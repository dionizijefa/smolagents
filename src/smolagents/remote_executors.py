from __future__ import annotations
#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
from dataclasses import dataclass, field
import inspect
import io
import json
import os
from pathlib import Path
import pickle
import re
import secrets
import shutil
import subprocess
import tarfile
import tempfile
import time
from contextlib import closing
from io import BytesIO
from textwrap import dedent
from typing import Any, List, Optional

import PIL.Image
import docker
import requests
from requests.exceptions import RequestException

from .default_tools import FinalAnswerTool
from .local_python_executor import CodeOutput, PythonExecutor
from .monitoring import LogLevel
from .tools import Tool, get_tools_definition_code
from .utils import AgentError


__all__ = ["E2BExecutor", "ModalExecutor", "DockerExecutor", "WasmExecutor", "LocalDockerExecutor"]

@dataclass
class ExecutionError:
    traceback: str


@dataclass
class ExecutionLogs:
    stdout: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    output: str = ""
    error: Optional[ExecutionError] = None
    logs: ExecutionLogs = field(default_factory=ExecutionLogs)



try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class RemotePythonExecutor(PythonExecutor):
    FINAL_ANSWER_EXCEPTION = "FinalAnswerException"

    def __init__(self, additional_imports: list[str], logger):
        self.additional_imports = additional_imports
        self.logger = logger
        self.logger.log("Initializing executor, hold on...")
        self.installed_packages = []

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute code, return the result and output, also determining if
        the result is the final answer.
        """
        raise NotImplementedError

    def send_tools(self, tools: dict[str, Tool]):
        if "final_answer" in tools:
            self._patch_final_answer_with_exception(tools["final_answer"])
        # Install tool packages
        packages_to_install = {
            pkg
            for tool in tools.values()
            for pkg in tool.to_dict()["requirements"]
            if pkg not in self.installed_packages + ["smolagents"]
        }
        if packages_to_install:
            self.installed_packages += self.install_packages(list(packages_to_install))
        # Get tool definitions
        code = get_tools_definition_code(tools)
        if code:
            code_output = self.run_code_raise_errors(code)
            self.logger.log(code_output.logs)

    def send_variables(self, variables: dict[str, Any]):
        """
        Send variables to the kernel namespace using pickle.
        """
        if not variables:
            return
        pickled_vars = base64.b64encode(pickle.dumps(variables)).decode()
        code = f"""
import pickle, base64
vars_dict = pickle.loads(base64.b64decode('{pickled_vars}'))
locals().update(vars_dict)
"""
        self.run_code_raise_errors(code)

    def __call__(self, code_action: str) -> CodeOutput:
        """Run the code and determine if it is the final answer."""
        return self.run_code_raise_errors(code_action)

    def install_packages(self, additional_imports: list[str]):
        if additional_imports:
            code_output = self.run_code_raise_errors(f"!pip install {' '.join(additional_imports)}")
            self.logger.log(code_output.logs)
        return additional_imports

    def _patch_final_answer_with_exception(self, final_answer_tool: FinalAnswerTool):
        """Patch the FinalAnswerTool to raise an exception.

        This is necessary because the remote executors
        rely on the FinalAnswerTool to detect the final answer.
        It modifies the `forward` method of the FinalAnswerTool to raise
        a `FinalAnswerException` with the final answer as a pickled value.
        This allows the executor to catch this exception and return the final answer.

        Args:
            final_answer_tool (`FinalAnswerTool`): FinalAnswerTool instance to patch.
        """

        # Create a new class that inherits from the original FinalAnswerTool
        class _FinalAnswerTool(final_answer_tool.__class__):
            pass

        # Add a new forward method that raises the FinalAnswerException
        # - Define the new forward method function
        def forward(self, *args, **kwargs) -> Any:
            import base64
            import pickle

            class FinalAnswerException(Exception):
                def __init__(self, value):
                    self.value = value

            raise FinalAnswerException(base64.b64encode(pickle.dumps(self._forward(*args, **kwargs))).decode())

        # - Set the new forward method function to the _FinalAnswerTool class
        _FinalAnswerTool.forward = forward

        # Rename the original forward method to _forward
        # - Get the original forward method function from the final_answer_tool instance
        original_forward_function = final_answer_tool.forward.__func__
        # - Set the new _forward method function to the _FinalAnswerTool class
        _FinalAnswerTool._forward = original_forward_function
        # - Update the source code of the new forward method to match the original but with the new name
        _FinalAnswerTool._forward.__source__ = inspect.getsource(original_forward_function).replace(
            "def forward(", "def _forward("
        )

        # Set the new class as the class of the final_answer_tool instance
        final_answer_tool.__class__ = _FinalAnswerTool


class E2BExecutor(RemotePythonExecutor):
    """
    Executes Python code using E2B.

    Args:
        additional_imports (`list[str]`): Additional imports to install.
        logger (`Logger`): Logger to use.
        **kwargs: Additional arguments to pass to the E2B Sandbox.
    """

    def __init__(self, additional_imports: list[str], logger, **kwargs):
        super().__init__(additional_imports, logger)
        try:
            from e2b_code_interpreter import Sandbox
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'e2b' extra to use E2BExecutor: `pip install 'smolagents[e2b]'`"""
            )
        # Support both e2b v1 and v2 constructors
        # v2 exposes Sandbox.create(...), while v1 uses Sandbox(...)
        if hasattr(Sandbox, "create"):
            self.sandbox = Sandbox.create(**kwargs)
        else:
            self.sandbox = Sandbox(**kwargs)
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("E2B is running", level=LogLevel.INFO)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        execution = self.sandbox.run_code(code)
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])

        # Handle errors
        if execution.error:
            # Check if the error is a FinalAnswerException
            if execution.error.name == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                final_answer = pickle.loads(base64.b64decode(execution.error.value))
                return CodeOutput(output=final_answer, logs=execution_logs, is_final_answer=True)

            # Construct error message
            error_message = (
                f"{execution_logs}\n"
                f"Executing code yielded an error:\n"
                f"{execution.error.name}\n"
                f"{execution.error.value}\n"
                f"{execution.error.traceback}"
            )
            raise AgentError(error_message, self.logger)

        # Handle results
        if not execution.results:
            return CodeOutput(output=None, logs=execution_logs, is_final_answer=False)

        for result in execution.results:
            if not result.is_main_result:
                continue
            # Handle image outputs
            for attribute_name in ["jpeg", "png"]:
                img_data = getattr(result, attribute_name, None)
                if img_data is not None:
                    decoded_bytes = base64.b64decode(img_data.encode("utf-8"))
                    return CodeOutput(
                        output=PIL.Image.open(BytesIO(decoded_bytes)), logs=execution_logs, is_final_answer=False
                    )
            # Handle other data formats
            for attribute_name in [
                "chart",
                "data",
                "html",
                "javascript",
                "json",
                "latex",
                "markdown",
                "pdf",
                "svg",
                "text",
            ]:
                data = getattr(result, attribute_name, None)
                if data is not None:
                    return CodeOutput(output=data, logs=execution_logs, is_final_answer=False)
        # If no main result found, return None
        return CodeOutput(output=None, logs=execution_logs, is_final_answer=False)

    def cleanup(self):
        """Clean up the E2B sandbox and resources."""
        try:
            if hasattr(self, "sandbox"):
                self.logger.log("Shutting down sandbox...", level=LogLevel.INFO)
                self.sandbox.kill()
                self.logger.log("Sandbox cleanup completed", level=LogLevel.INFO)
                del self.sandbox
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")


def _websocket_send_execute_request(code: str, ws) -> str:
    """Send code execution request to kernel."""
    import uuid

    # Generate a unique message ID
    msg_id = str(uuid.uuid4())

    # Create execute request
    execute_request = {
        "header": {
            "msg_id": msg_id,
            "username": "anonymous",
            "session": str(uuid.uuid4()),
            "msg_type": "execute_request",
            "version": "5.0",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
        },
    }

    ws.send(json.dumps(execute_request))
    return msg_id


def _websocket_run_code_raise_errors(code: str, ws, logger) -> CodeOutput:
    """Run code over a websocket."""
    try:
        # Send execute request
        msg_id = _websocket_send_execute_request(code, ws)

        # Collect output and results
        outputs = []
        result = None
        is_final_answer = False

        while True:
            msg = json.loads(ws.recv())
            parent_msg_id = msg.get("parent_header", {}).get("msg_id")
            # Skip unrelated messages
            if parent_msg_id != msg_id:
                continue
            msg_type = msg.get("msg_type", "")
            msg_content = msg.get("content", {})
            if msg_type == "stream":
                outputs.append(msg_content["text"])
            elif msg_type == "execute_result":
                result = msg_content["data"].get("text/plain", None)
            elif msg_type == "error":
                if msg_content.get("ename", "") == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                    result = pickle.loads(base64.b64decode(msg_content.get("evalue", "")))
                    is_final_answer = True
                else:
                    raise AgentError("\n".join(msg_content.get("traceback", [])), logger)
            elif msg_type == "status" and msg_content["execution_state"] == "idle":
                break

        return CodeOutput(output=result, logs="".join(outputs), is_final_answer=is_final_answer)

    except Exception as e:
        logger.log_error(f"Code execution failed: {e}")
        raise


def _create_kernel_http(crate_kernel_endpoint: str, logger) -> str:
    """Create kernel using http."""

    r = requests.post(crate_kernel_endpoint)
    if r.status_code != 201:
        error_details = {
            "status_code": r.status_code,
            "headers": dict(r.headers),
            "url": r.url,
            "body": r.text,
            "request_method": r.request.method,
            "request_headers": dict(r.request.headers),
            "request_body": r.request.body,
        }
        logger.log_error(f"Failed to create kernel. Details: {json.dumps(error_details, indent=2)}")
        raise RuntimeError(f"Failed to create kernel: Status {r.status_code}\nResponse: {r.text}") from None
    return r.json()["id"]


class DockerExecutor(RemotePythonExecutor):
    """
    Executes Python code using Jupyter Kernel Gateway in a Docker container.
    Adds host<->container bind mounts for inputs/outputs.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        host: str = "127.0.0.1",
        port: int = 8888,
        image_name: str = "jupyter-kernel",
        build_new_image: bool = True,
        container_run_kwargs: dict[str, Any] | None = None,
        dockerfile_content: str | None = None,

        # NEW: mount-related convenience params
        data_path: Optional[str] = None,         # host path to mount at /workspace/data (ro)
        reference_path: Optional[str] = None,    # host path to mount at /workspace/reference (ro)
        output_path: Optional[str] = None,       # host path to mount at /workspace/output (rw)
        results_path: Optional[str] = None,      # host path to mount at /workspace/results (rw)
        extra_mounts: Optional[list[tuple[str, str, str]]] = None,
        # ^ list of (host_path, container_path, mode) e.g. [("/host/dir","/mnt/x","ro")]
        ensure_dirs: bool = True,                # create host output/results dirs if missing
    ):
        """
        Initialize the Docker-based Jupyter Kernel Gateway executor.

        Args:
            additional_imports: Additional imports to install inside the kernel env.
            logger: Logger to use (expects .log / .log_error with level).
            host: Host to bind to for Kernel Gateway.
            port: Port to bind to for Kernel Gateway.
            image_name: Docker image name/tag to use or build.
            build_new_image: Force rebuild even if image exists.
            container_run_kwargs: Extra kwargs passed to docker.containers.run.
            dockerfile_content: Custom Dockerfile content.

            data_path: Host path mounted read-only at /workspace/data.
            reference_path: Host path mounted read-only at /workspace/reference.
            output_path: Host path mounted read-write at /workspace/output.
            results_path: Host path mounted read-write at /workspace/results.
            extra_mounts: Additional (host, container, mode) mounts.
            ensure_dirs: If True, create output/results dirs on host.
        """
        super().__init__(additional_imports, logger)

        # --- imports guarded for nicer error message
        try:
            import docker
            from websocket import create_connection
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'docker' extra to use DockerExecutor: `pip install 'smolagents[docker]'`"
            )

        self.host = host
        self.port = port
        self.image_name = image_name
        self.logger = logger

        self.dockerfile_content = dockerfile_content or dedent(
            """\
            FROM python:3.12-bullseye

            RUN pip install --no-cache-dir jupyter_kernel_gateway jupyter_client ipykernel

            EXPOSE 8888
            CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin=*"]
            """
        )

        # --- prepare mounts
        def _p(p: Optional[str]) -> Optional[Path]:
            return Path(p).expanduser().resolve() if p else None

        data_p      = _p(data_path)
        ref_p       = _p(reference_path)
        out_p       = _p(output_path)
        res_p       = _p(results_path)

        if ensure_dirs:
            # inputs are optional; outputs we often want to exist
            for d in [out_p, res_p]:
                if d:
                    d.mkdir(parents=True, exist_ok=True)

        self._mounts = []  # (host_path: Path, container_path: str, mode: str)

        if data_p:
            self._mounts.append((data_p, "/workspace/data", "ro"))
        if ref_p:
            self._mounts.append((ref_p, "/workspace/reference", "ro"))
        if out_p:
            self._mounts.append((out_p, "/workspace/output", "rw"))
        if res_p:
            self._mounts.append((res_p, "/workspace/results", "rw"))

        if extra_mounts:
            for host, container, mode in extra_mounts:
                self._mounts.append((_p(host), container, mode))

        # --- Initialize Docker
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError("Could not connect to Docker daemon: make sure Docker is running.") from e

        # --- Build and start container
        try:
            # Build (if needed)
            if not build_new_image:
                try:
                    self.client.images.get(self.image_name)
                    self.logger.log(f"Using existing Docker image: {self.image_name}", level=LogLevel.INFO)
                except self.client.api.images().__class__.ImageNotFound if hasattr(self.client, "api") else Exception:  # defensive
                    self.logger.log(f"Image {self.image_name} not found, building...", level=LogLevel.INFO)
                    build_new_image = True

            if build_new_image:
                self.logger.log(f"Building Docker image {self.image_name}...", level=LogLevel.INFO)
                dockerfile_obj = BytesIO(self.dockerfile_content.encode("utf-8"))
                _, build_logs = self.client.images.build(fileobj=dockerfile_obj, tag=self.image_name, rm=True, forcerm=True)
                for log_chunk in build_logs:
                    if log_message := log_chunk.get("stream", "").rstrip():
                        self.logger.log(log_message, level=LogLevel.DEBUG)

            self.logger.log(f"Starting container on {host}:{port}...", level=LogLevel.INFO)

            # Base container kwargs
            container_kwargs: dict[str, Any] = {}
            if container_run_kwargs:
                container_kwargs.update(container_run_kwargs)

            # --- Ensure Kernel Gateway port mapping + detach
            if not isinstance(container_kwargs.get("ports"), dict):
                container_kwargs["ports"] = {}
            container_kwargs["ports"]["8888/tcp"] = (host, port)
            container_kwargs["detach"] = True

            # --- Merge volumes
            # docker-py expects: {"host_path": {"bind": "/in/container", "mode": "ro|rw"}, ...}
            volumes = container_kwargs.get("volumes", {})
            for host_path, container_path, mode in self._mounts:
                if host_path is None:
                    continue
                # Docker Desktop on Windows: ensure drive-letter style is acceptable; Path.resolve() usually OK.
                volumes[str(host_path)] = {"bind": container_path, "mode": mode}
            if volumes:
                container_kwargs["volumes"] = volumes

            # You may also add extra_hosts if needed:
            # container_kwargs.setdefault("extra_hosts", {})["host.docker.internal"] = "host-gateway"

            self.container = self.client.containers.run(self.image_name, **container_kwargs)

            # Wait for running
            retries = 0
            while self.container.status != "running" and retries < 10:
                self.logger.log(f"Container status: {self.container.status}, waiting...", level=LogLevel.INFO)
                time.sleep(0.8)
                self.container.reload()
                retries += 1

            self.base_url = f"http://{host}:{port}"

            # Wait for Jupyter
            self._wait_for_server()

            # Create new kernel via HTTP
            self.kernel_id = _create_kernel_http(f"{self.base_url}/api/kernels", self.logger)

            ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels"
            self.ws = create_connection(ws_url)

            self.installed_packages = self.install_packages(additional_imports)
            self.logger.log(
                f"Container {self.container.short_id} is running with kernel {self.kernel_id}",
                level=LogLevel.INFO,
            )

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}") from e

    def run_code_raise_errors(self, code: str) -> "CodeOutput":
        return _websocket_run_code_raise_errors(code, self.ws, self.logger)

    def cleanup(self):
        """Clean up the Docker container and resources."""
        try:
            if hasattr(self, "container"):
                self.logger.log(f"Stopping and removing container {self.container.short_id}...", level=LogLevel.INFO)
                self.container.stop()
                self.container.remove()
                self.logger.log("Container cleanup completed", level=LogLevel.INFO)
                del self.container
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def _wait_for_server(self):
        retries = 0
        jupyter_ready = False
        while not jupyter_ready and retries < 20:
            try:
                r = requests.get(f"{self.base_url}/api/kernelspecs", timeout=2)
                if r.status_code == 200:
                    jupyter_ready = True
                else:
                    self.logger.log("Jupyter not ready, waiting...", level=LogLevel.INFO)
            except requests.RequestException:
                self.logger.log("Jupyter not ready, waiting...", level=LogLevel.INFO)
            if not jupyter_ready:
                time.sleep(0.8)
                retries += 1


class ModalExecutor(RemotePythonExecutor):
    """
    Executes Python code using Modal.

    Args:
        additional_imports: Additional imports to install.
        logger (`Logger`): Logger to use for output and errors.
        app_name (`str`): App name.
        port (`int`): Port for jupyter to bind to.
        create_kwargs (`dict`, optional): Keyword arguments to pass to creating the sandbox. See
            `modal.Sandbox.create` [docs](https://modal.com/docs/reference/modal.Sandbox#create) for all the
            keyword arguments.
    """

    _ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        app_name: str = "smolagent-executor",
        port: int = 8888,
        create_kwargs: Optional[dict] = None,
    ):
        super().__init__(additional_imports, logger)
        self.port = port
        try:
            import modal
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'modal' extra to use ModalExecutor: `pip install 'smolagents[modal]'`"""
            )

        if create_kwargs is None:
            create_kwargs = {}

        create_kwargs = {
            "image": modal.Image.debian_slim().uv_pip_install("jupyter_kernel_gateway", "ipykernel"),
            "timeout": 60 * 5,
            **create_kwargs,
        }

        if "app" not in create_kwargs:
            create_kwargs["app"] = modal.App.lookup(app_name, create_if_missing=True)

        if "encrypted_ports" not in create_kwargs:
            create_kwargs["encrypted_ports"] = [port]
        else:
            create_kwargs["encrypted_ports"] = create_kwargs["encrypted_ports"] + [port]

        token = secrets.token_urlsafe(16)
        default_secrets = [modal.Secret.from_dict({"KG_AUTH_TOKEN": token})]

        if "secrets" not in create_kwargs:
            create_kwargs["secrets"] = default_secrets
        else:
            create_kwargs["secrets"] = create_kwargs["secrets"] + default_secrets

        entrypoint = [
            "jupyter",
            "kernelgateway",
            "--KernelGatewayApp.ip='0.0.0.0'",
            f"--KernelGatewayApp.port={port}",
            "--KernelGatewayApp.allow_origin='*'",
        ]

        self.logger.log("Starting Modal sandbox", level=LogLevel.INFO)
        self.sandbox = modal.Sandbox.create(
            *entrypoint,
            **create_kwargs,
        )

        tunnel = self.sandbox.tunnels()[port]
        self.logger.log(f"Waiting for Modal sandbox on {tunnel.host}:{port}", level=LogLevel.INFO)
        self._wait_for_server(tunnel.host, token)

        self.logger.log("Starting Jupyter kernel", level=LogLevel.INFO)
        kernel_id = _create_kernel_http(f"https://{tunnel.host}/api/kernels?token={token}", logger)
        self.ws_url = f"wss://{tunnel.host}/api/kernels/{kernel_id}/channels?token={token}"
        self.installed_packages = self.install_packages(additional_imports)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        from websocket import create_connection

        with closing(create_connection(self.ws_url)) as ws:
            return _websocket_run_code_raise_errors(code, ws, self.logger)

    def cleanup(self):
        if hasattr(self, "sandbox"):
            self.sandbox.terminate()

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def _wait_for_server(self, host: str, token: str):
        """Wait for server to start up."""
        n_retries = 0
        while True:
            try:
                resp = requests.get(f"https://{host}/api/kernelspecs?token={token}")
                if resp.status_code == 200:
                    break
            except RequestException:
                n_retries += 1
                if n_retries % 10 == 0:
                    self.logger.log("Waiting for server to startup, retrying...", level=LogLevel.INFO)
                if n_retries > 60:
                    raise RuntimeError("Unable to connect to sandbox")
                time.sleep(1.0)

    @classmethod
    def _strip_ansi_colors(cls, text: str) -> str:
        """Remove ansi colors from text."""
        return cls._ANSI_ESCAPE.sub("", text)


class WasmExecutor(RemotePythonExecutor):
    """
    Remote Python code executor in a sandboxed WebAssembly environment powered by Pyodide and Deno.

    This executor combines Deno's secure runtime with Pyodide's WebAssembly‑compiled Python interpreter to deliver s
    trong isolation guarantees while enabling full Python execution.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install in the Pyodide environment.
        logger (`Logger`): Logger to use for output and errors.
        deno_path (`str`, optional): Path to the Deno executable. If not provided, will use "deno" from PATH.
        deno_permissions (`list[str]`, optional): List of permissions to grant to the Deno runtime.
            Default is minimal permissions needed for execution.
        timeout (`int`, optional): Timeout in seconds for code execution. Default is 60 seconds.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        deno_path: str = "deno",
        deno_permissions: list[str] | None = None,
        timeout: int = 60,
    ):
        super().__init__(additional_imports, logger)

        # Check if Deno is installed
        try:
            subprocess.run([deno_path, "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "Deno is not installed or not found in PATH. Please install Deno from https://deno.land/"
            )

        self.deno_path = deno_path
        self.timeout = timeout

        # Default minimal permissions needed
        if deno_permissions is None:
            # Use minimal permissions for Deno execution
            home_dir = os.getenv("HOME")
            deno_permissions = [
                "allow-net="
                + ",".join(
                    [
                        "0.0.0.0:8000",  # allow requests to the local server
                        "cdn.jsdelivr.net:443",  # allow loading pyodide packages
                        "pypi.org:443,files.pythonhosted.org:443",  # allow pyodide install packages from PyPI
                    ]
                ),
                f"allow-read={home_dir}/.cache/deno",
                f"allow-write={home_dir}/.cache/deno",
            ]
        self.deno_permissions = [f"--{perm}" for perm in deno_permissions]

        # Create the Deno JavaScript runner file
        self._create_deno_runner()

        # Install additional packages
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("WasmExecutor is running", level=LogLevel.INFO)

    def _create_deno_runner(self):
        """Create the Deno JavaScript file that will run Pyodide and execute Python code."""
        self.runner_dir = tempfile.mkdtemp(prefix="pyodide_deno_")
        self.runner_path = os.path.join(self.runner_dir, "pyodide_runner.js")

        # Create the JavaScript runner file
        with open(self.runner_path, "w") as f:
            f.write(self.JS_CODE)

        # Start the Deno server
        self._start_deno_server()

    def _start_deno_server(self):
        """Start the Deno server that will run our JavaScript code."""
        cmd = [self.deno_path, "run"] + self.deno_permissions + [self.runner_path]

        # Start the server process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for the server to start
        time.sleep(2)  # Give the server time to start

        # Check if the server started successfully
        if self.server_process.poll() is not None:
            stderr = self.server_process.stderr.read()
            raise RuntimeError(f"Failed to start Deno server: {stderr}")

        self.server_url = "http://localhost:8000"  # TODO: Another port?

        # Test the connection
        try:
            response = requests.get(self.server_url)
            if response.status_code != 200:
                raise RuntimeError(f"Server responded with status code {response.status_code}: {response.text}")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to Deno server: {e}")

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute Python code in the Pyodide environment and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        try:
            # Prepare the request payload
            payload = {
                "code": code,
                "packages": self.installed_packages,
            }

            # Send the request to the Deno server
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)

            if response.status_code != 200:
                raise AgentError(f"Server error: {response.text}", self.logger)

            result = None
            is_final_answer = False

            # Parse the response
            result_data = response.json()

            # Process the result
            if result_data.get("result"):
                result = result_data.get("result")
            # Check for execution errors
            elif result_data.get("error"):
                error = result_data["error"]
                if (
                    error.get("pythonExceptionType") == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION
                    and "pythonExceptionValue" in error
                ):
                    result = pickle.loads(base64.b64decode(error["pythonExceptionValue"]))
                    is_final_answer = True
                else:
                    error_message = f"{error.get('name', 'Error')}: {error.get('message', 'Unknown error')}"
                    if "stack" in error:
                        error_message += f"\n{error['stack']}"
                    raise AgentError(error_message, self.logger)

            # Get the execution logs
            execution_logs = result_data.get("stdout", "")

            # Handle image results
            if isinstance(result, dict) and result.get("type") == "image":
                image_data = result.get("data", "")
                decoded_bytes = base64.b64decode(image_data.encode("utf-8"))
                return PIL.Image.open(BytesIO(decoded_bytes)), execution_logs

            return CodeOutput(output=result, logs=execution_logs, is_final_answer=is_final_answer)

        except requests.RequestException as e:
            raise AgentError(f"Failed to communicate with Deno server: {e}", self.logger)

    def install_packages(self, additional_imports: list[str]) -> list[str]:
        """
        Install additional Python packages in the Pyodide environment.

        Args:
            additional_imports (`list[str]`): Package names to install.

        Returns:
            list[str]: Installed packages.
        """
        # In Pyodide, we don't actually install packages here, but we keep track of them
        # to load them when executing code
        # TODO: Install  here instead?
        self.logger.log(f"Adding packages to load: {', '.join(additional_imports)}", level=LogLevel.INFO)
        return additional_imports

    def cleanup(self):
        """Clean up resources used by the executor."""
        if hasattr(self, "server_process") and self.server_process:
            self.logger.log("Stopping Deno server...", level=LogLevel.INFO)
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

        # Remove the temporary directory
        if hasattr(self, "runner_dir") and os.path.exists(self.runner_dir):
            import shutil

            shutil.rmtree(self.runner_dir)

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    JS_CODE = dedent("""\
        // pyodide_runner.js - Runs Python code in Pyodide within Deno
        import { serve } from "https://deno.land/std/http/server.ts";
        import { loadPyodide } from "npm:pyodide";

        // Initialize Pyodide instance
        const pyodidePromise = loadPyodide();

        // Function to execute Python code and return the result
        async function executePythonCode(code) {
          const pyodide = await pyodidePromise;

          // Create a capture for stdout
          pyodide.runPython(`
            import sys
            import io
            sys.stdout = io.StringIO()
          `);

          // Execute the code and capture any errors
          let result = null;
          let error = null;
          let stdout = "";

          try {
            // Execute the code
            result = await pyodide.runPythonAsync(code);

            // Get captured stdout
            stdout = pyodide.runPython("sys.stdout.getvalue()");
          } catch (e) {
            error = {
              name: e.constructor.name,
              message: e.message,
              stack: e.stack
            };

            // Extract Python exception details
            if (e.constructor.name === "PythonError") {
              // Get the Python exception type from the error message: at the end of the traceback
              const errorMatch = e.message.match(/\\n([^:]+Exception): /);
              if (errorMatch) {
                error.pythonExceptionType = errorMatch[1].split(".").pop();
              }

              // If the error is a FinalAnswerException, extract its the encoded value
              if (error.pythonExceptionType === "FinalAnswerException") {
                // Extract the base64 encoded value from the error message
                const valueMatch = e.message.match(/FinalAnswerException: (.*?)(?:\\n|$)/);
                if (valueMatch) {
                  error.pythonExceptionValue = valueMatch[1];
                }
              }
            }
          }

          return {
            result,
            stdout,
            error
          };
        }

        // Start a simple HTTP server to receive code execution requests
        //const port = 8765;
        //console.log(`Starting Pyodide server on port ${port}`);

        serve(async (req) => {
          if (req.method === "POST") {
            try {
              const body = await req.json();
              const { code, packages = [] } = body;

              // Load any requested packages
              if (packages && packages.length > 0) {
                const pyodide = await pyodidePromise;
                //await pyodide.loadPackagesFromImports(code);
                await pyodide.loadPackage("micropip");
                const micropip = pyodide.pyimport("micropip");
                try {
                  await micropip.install(packages);
                } catch (e) {
                  console.error(`Failed to load package ${pkg}: ${e.message}`);
                }
              }

              const result = await executePythonCode(code);
              return new Response(JSON.stringify(result), {
                headers: { "Content-Type": "application/json" }
              });
            } catch (e) {
              return new Response(JSON.stringify({ error: e.message }), {
                status: 500,
                headers: { "Content-Type": "application/json" }
              });
            }
          }

          return new Response("Pyodide-Deno Executor is running. Send POST requests with code to execute.", {
            headers: { "Content-Type": "text/plain" }
          });
        });
        """)

class LocalDockerExecutor:
    """
    Local Docker executor that:
      - builds (or reuses) a micromamba-based image,
      - copies inputs (data/, reference/) into /workspace,
      - runs Python code inside the container,
      - copies /workspace/output and /workspace/results back to the host.
    """

    def __init__(
        self,
        *,
        image_name: str = "micromamba-executor:latest",
        build_new_image: bool = True,
        dockerfile_content: Optional[str] = None,
        container_run_kwargs: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,  # expects .info/.debug/.error or print-like
        python_bin: str = "/opt/conda/bin/python",  # micromamba base default
    ):
        self.client = docker.from_env()
        self.image_name = image_name
        self.build_new_image = build_new_image
        self.python_bin = python_bin
        self.container_run_kwargs = container_run_kwargs or {}
        self.container = None
        self.logger = logger
        self._dockerfile_content = dockerfile_content or dedent(
            """\
            FROM mambaorg/micromamba

            ARG MAMBA_DOCKERFILE_ACTIVATE=1
            WORKDIR /workspace

            # minimal, pin Python; add what you need here (pip, uv, etc.)
            RUN micromamba install -y -n base -c conda-forge python=3.13 && \
                micromamba clean -a -y

            # Optionally: create output/results dirs to ensure they exist
            RUN mkdir -p /workspace/output /workspace/results /workspace/data /workspace/reference
            """
        )

        self._ensure_image()

    # ---------- Logging helpers ----------
    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            fn = getattr(self.logger, level, None)
            if callable(fn):
                fn(msg)
                return
        print(msg)

    # ---------- Image build / ensure ----------
    def _ensure_image(self):
        image_found = False
        if not self.build_new_image:
            try:
                self.client.images.get(self.image_name)
                image_found = True
                self._log(f"Using existing image: {self.image_name}")
            except docker.errors.ImageNotFound:
                self._log(f"Image {self.image_name} not found; will build.")

        if self.build_new_image or not image_found:
            self._log(f"Building Docker image {self.image_name}...")
            dockerfile_bytes = io.BytesIO(self._dockerfile_content.encode("utf-8"))
            image, build_logs = self.client.images.build(
                fileobj=dockerfile_bytes,
                tag=self.image_name,
                rm=True,
                forcerm=True,
                pull=False,
            )
            for line in build_logs:
                stream = line.get("stream")
                if stream:
                    s = stream.strip()
                    if s:
                        self._log(s, level="debug")

    # ---------- Container lifecycle ----------
    def _start_container(self, env: Optional[Dict[str, str]] = None):
        # Safe defaults; caller can override via container_run_kwargs
        run_kwargs = dict(
            command="tail -f /dev/null",
            detach=True,
            tty=True,
            working_dir="/workspace",
            # Drop capabilities by default (you can loosen if needed)
            cap_drop=["ALL"],
            environment=env or {},
            # Add host-gateway if you need to call back to host services
            extra_hosts={"host.docker.internal": "host-gateway"},
        )
        run_kwargs.update(self.container_run_kwargs or {})

        self.container = self.client.containers.run(self.image_name, **run_kwargs)
        # Wait a moment and confirm it's up
        retries = 0
        while retries < 10:
            self.container.reload()
            if self.container.status == "running":
                break
            time.sleep(0.4)
            retries += 1
        if self.container.status != "running":
            raise RuntimeError(f"Container failed to start (status={self.container.status}).")
        self._log(f"Container started: {self.container.short_id}")

    def cleanup(self):
        if self.container:
            try:
                self._log(f"Stopping container {self.container.short_id}...")
                self.container.stop(timeout=5)
            except docker.errors.NotFound:
                pass
            except Exception as e:
                self._log(f"Error during container stop: {e}", level="error")
            try:
                self._log(f"Removing container {self.container.short_id}...")
                self.container.remove(force=True)
            except Exception as e:
                self._log(f"Error during container remove: {e}", level="error")
            finally:
                self.container = None

    # ---------- Tar copy helpers ----------
    def _copy_dir_to_container(self, host_dir: Path, container_dest: Path):
        if not host_dir.exists() or not host_dir.is_dir():
            return
        # Create a tar with the directory contents (as a top-level folder named like container_dest.name)
        archive_stream = io.BytesIO()
        with tarfile.open(fileobj=archive_stream, mode="w") as tar:
            tar.add(str(host_dir), arcname=container_dest.name)
        archive_stream.seek(0)
        # Ensure parent exists and remove previous target
        self.container.exec_run(["mkdir", "-p", str(container_dest.parent)])
        self.container.exec_run(["rm", "-rf", str(container_dest)])
        self.container.put_archive(str(container_dest.parent), archive_stream.getvalue())

    def _copy_dir_from_container(self, container_src: Path, host_dest: Path):
        try:
            stream, _ = self.container.get_archive(str(container_src))
        except docker.errors.NotFound:
            return

        tmpdir = Path(tempfile.mkdtemp(prefix="dockercopy_"))
        try:
            buf = io.BytesIO()
            for chunk in stream:
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf, mode="r") as tar:
                # SECURITY NOTE: tar.extractall is used inside tmp dir which we control.
                tar.extractall(path=tmpdir)

            extracted_root = tmpdir / container_src.name
            host_dest = host_dest.resolve()
            if host_dest.exists():
                shutil.rmtree(host_dest)
            if extracted_root.is_dir():
                shutil.copytree(extracted_root, host_dest)
            elif extracted_root.exists():
                host_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(extracted_root, host_dest)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ---------- Public API ----------
    def run_code(
        self,
        code: str,
        *,
        task_data_path: Optional[str] = None,   # host root with data/ and reference/
        output_path: Optional[str] = None,      # host dir to receive /workspace/output
        results_path: Optional[str] = None,     # host dir to receive /workspace/results
        env: Optional[Dict[str, str]] = None,
        log_file_dir: Optional[str] = None,     # where to write docker_stdout.log on host
        python_bin: Optional[str] = None,       # override interpreter inside container
        ensure_dirs: bool = True,
    ) -> ExecutionResult:
        """
        Execute Python code inside the container.

        Expected container paths:
          - /workspace/data        (copied from {task_data_path}/data if exists)
          - /workspace/reference   (copied from {task_data_path}/reference if exists)
          - /workspace/output      (copied back to output_path)
          - /workspace/results     (copied back to results_path)
        """
        # Expand/normalize host paths
        output_root = Path(output_path).expanduser().resolve() if output_path else None
        results_root = Path(results_path).expanduser().resolve() if results_path else None
        inputs_root = Path(task_data_path).expanduser().resolve() if task_data_path else None

        # Choose a sensible place for the log file
        log_dir_candidates = [p.parent for p in (output_root, results_root, inputs_root) if p is not None]
        log_dir = Path(log_file_dir).expanduser().resolve() if log_file_dir else (log_dir_candidates[0] if log_dir_candidates else Path.cwd())
        if ensure_dirs:
            log_dir.mkdir(parents=True, exist_ok=True)
            if output_root:
                output_root.mkdir(parents=True, exist_ok=True)
            if results_root:
                results_root.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "docker_stdout.log"

        # Start container
        if not self.container:
            self._start_container(env=env)

        # Copy inputs into container
        if inputs_root:
            self._copy_dir_to_container(inputs_root / "data", Path("/workspace/data"))
            self._copy_dir_to_container(inputs_root / "reference", Path("/workspace/reference"))

        # Ensure output/results dirs exist in container
        self.container.exec_run(["mkdir", "-p", "/workspace/output", "/workspace/results"])

        # Execute the code
        pybin = python_bin or self.python_bin
        try:
            exec_result = self.container.exec_run(
                cmd=[pybin, "-u", "-c", code],
                user="root",            # flip to non-root if you prefer
                environment=env or {},
                stream=True,
                demux=True,
            )

            out_chunks: List[str] = []
            with log_file_path.open("w", encoding="utf-8") as lf:
                for stdout_chunk, stderr_chunk in exec_result.output:
                    if stdout_chunk:
                        s = stdout_chunk.decode(errors="replace")
                        out_chunks.append(s)
                        lf.write(s)
                        lf.flush()
                    if stderr_chunk:
                        s = stderr_chunk.decode(errors="replace")
                        out_chunks.append(s)
                        lf.write(s)
                        lf.flush()

            combined = "".join(out_chunks)
            exit_code = exec_result.exit_code

            # Copy results back
            if output_root:
                self._copy_dir_from_container(Path("/workspace/output"), output_root)
            if results_root:
                self._copy_dir_from_container(Path("/workspace/results"), results_root)

            if exit_code != 0:
                return ExecutionResult(output=combined, error=ExecutionError(traceback=combined), logs=ExecutionLogs(stdout=[combined] if combined else []))
            return ExecutionResult(output=combined, logs=ExecutionLogs(stdout=[combined] if combined else []))

        except Exception as e:
            return ExecutionResult(error=ExecutionError(traceback=str(e)))
        # NOTE: no auto-cleanup here so you can inspect the container if needed.
        # Call .cleanup() explicitly when done.