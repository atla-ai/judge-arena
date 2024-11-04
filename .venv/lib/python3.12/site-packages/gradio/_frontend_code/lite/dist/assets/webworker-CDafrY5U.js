(function(){"use strict";var k=Object.defineProperty,P=(e,t,o)=>t in e?k(e,t,{enumerable:!0,configurable:!0,writable:!0,value:o}):e[t]=o,A=(e,t,o)=>(P(e,typeof t!="symbol"?t+"":t,o),o);function x(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}function h(e){if(typeof e!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(e))}function T(e,t){for(var o="",s=0,n=-1,i=0,r,a=0;a<=e.length;++a){if(a<e.length)r=e.charCodeAt(a);else{if(r===47)break;r=47}if(r===47){if(!(n===a-1||i===1))if(n!==a-1&&i===2){if(o.length<2||s!==2||o.charCodeAt(o.length-1)!==46||o.charCodeAt(o.length-2)!==46){if(o.length>2){var l=o.lastIndexOf("/");if(l!==o.length-1){l===-1?(o="",s=0):(o=o.slice(0,l),s=o.length-1-o.lastIndexOf("/")),n=a,i=0;continue}}else if(o.length===2||o.length===1){o="",s=0,n=a,i=0;continue}}t&&(o.length>0?o+="/..":o="..",s=2)}else o.length>0?o+="/"+e.slice(n+1,a):o=e.slice(n+1,a),s=a-n-1;n=a,i=0}else r===46&&i!==-1?++i:i=-1}return o}function D(e,t){var o=t.dir||t.root,s=t.base||(t.name||"")+(t.ext||"");return o?o===t.root?o+s:o+e+s:s}var u={resolve:function(){for(var e="",t=!1,o,s=arguments.length-1;s>=-1&&!t;s--){var n;s>=0?n=arguments[s]:(o===void 0&&(o=process.cwd()),n=o),h(n),n.length!==0&&(e=n+"/"+e,t=n.charCodeAt(0)===47)}return e=T(e,!t),t?e.length>0?"/"+e:"/":e.length>0?e:"."},normalize:function(e){if(h(e),e.length===0)return".";var t=e.charCodeAt(0)===47,o=e.charCodeAt(e.length-1)===47;return e=T(e,!t),e.length===0&&!t&&(e="."),e.length>0&&o&&(e+="/"),t?"/"+e:e},isAbsolute:function(e){return h(e),e.length>0&&e.charCodeAt(0)===47},join:function(){if(arguments.length===0)return".";for(var e,t=0;t<arguments.length;++t){var o=arguments[t];h(o),o.length>0&&(e===void 0?e=o:e+="/"+o)}return e===void 0?".":u.normalize(e)},relative:function(e,t){if(h(e),h(t),e===t||(e=u.resolve(e),t=u.resolve(t),e===t))return"";for(var o=1;o<e.length&&e.charCodeAt(o)===47;++o);for(var s=e.length,n=s-o,i=1;i<t.length&&t.charCodeAt(i)===47;++i);for(var r=t.length,a=r-i,l=n<a?n:a,d=-1,c=0;c<=l;++c){if(c===l){if(a>l){if(t.charCodeAt(i+c)===47)return t.slice(i+c+1);if(c===0)return t.slice(i+c)}else n>l&&(e.charCodeAt(o+c)===47?d=c:c===0&&(d=0));break}var f=e.charCodeAt(o+c),V=t.charCodeAt(i+c);if(f!==V)break;f===47&&(d=c)}var g="";for(c=o+d+1;c<=s;++c)(c===s||e.charCodeAt(c)===47)&&(g.length===0?g+="..":g+="/..");return g.length>0?g+t.slice(i+d):(i+=d,t.charCodeAt(i)===47&&++i,t.slice(i))},_makeLong:function(e){return e},dirname:function(e){if(h(e),e.length===0)return".";for(var t=e.charCodeAt(0),o=t===47,s=-1,n=!0,i=e.length-1;i>=1;--i)if(t=e.charCodeAt(i),t===47){if(!n){s=i;break}}else n=!1;return s===-1?o?"/":".":o&&s===1?"//":e.slice(0,s)},basename:function(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');h(e);var o=0,s=-1,n=!0,i;if(t!==void 0&&t.length>0&&t.length<=e.length){if(t.length===e.length&&t===e)return"";var r=t.length-1,a=-1;for(i=e.length-1;i>=0;--i){var l=e.charCodeAt(i);if(l===47){if(!n){o=i+1;break}}else a===-1&&(n=!1,a=i+1),r>=0&&(l===t.charCodeAt(r)?--r===-1&&(s=i):(r=-1,s=a))}return o===s?s=a:s===-1&&(s=e.length),e.slice(o,s)}else{for(i=e.length-1;i>=0;--i)if(e.charCodeAt(i)===47){if(!n){o=i+1;break}}else s===-1&&(n=!1,s=i+1);return s===-1?"":e.slice(o,s)}},extname:function(e){h(e);for(var t=-1,o=0,s=-1,n=!0,i=0,r=e.length-1;r>=0;--r){var a=e.charCodeAt(r);if(a===47){if(!n){o=r+1;break}continue}s===-1&&(n=!1,s=r+1),a===46?t===-1?t=r:i!==1&&(i=1):t!==-1&&(i=-1)}return t===-1||s===-1||i===0||i===1&&t===s-1&&t===o+1?"":e.slice(t,s)},format:function(e){if(e===null||typeof e!="object")throw new TypeError('The "pathObject" argument must be of type Object. Received type '+typeof e);return D("/",e)},parse:function(e){h(e);var t={root:"",dir:"",base:"",ext:"",name:""};if(e.length===0)return t;var o=e.charCodeAt(0),s=o===47,n;s?(t.root="/",n=1):n=0;for(var i=-1,r=0,a=-1,l=!0,d=e.length-1,c=0;d>=n;--d){if(o=e.charCodeAt(d),o===47){if(!l){r=d+1;break}continue}a===-1&&(l=!1,a=d+1),o===46?i===-1?i=d:c!==1&&(c=1):i!==-1&&(c=-1)}return i===-1||a===-1||c===0||c===1&&i===a-1&&i===r+1?a!==-1&&(r===0&&s?t.base=t.name=e.slice(1,a):t.base=t.name=e.slice(r,a)):(r===0&&s?(t.name=e.slice(1,i),t.base=e.slice(1,a)):(t.name=e.slice(r,i),t.base=e.slice(r,a)),t.ext=e.slice(i,a)),r>0?t.dir=e.slice(0,r-1):s&&(t.dir="/"),t},sep:"/",delimiter:":",win32:null,posix:null};u.posix=u;var M=u;const y=x(M),F="/home/pyodide",b=e=>`${F}/${e}`,m=(e,t)=>(y.normalize(t),y.resolve(b(e),t));function S(e,t){const o=y.normalize(t),s=y.dirname(o).split("/"),n=[];for(const i of s){n.push(i);const r=n.join("/");if(e.FS.analyzePath(r).exists){if(e.FS.isDir(r))throw new Error(`"${r}" already exists and is not a directory.`);continue}try{e.FS.mkdir(r)}catch(a){throw console.error(`Failed to create a directory "${r}"`),a}}}function E(e,t,o,s){S(e,t),e.FS.writeFile(t,o,s)}function H(e,t,o){S(e,o),e.FS.rename(t,o)}function U(e){e.forEach(t=>{let o;try{o=new URL(t)}catch{return}if(o.protocol==="emfs:"||o.protocol==="file:")throw new Error(`"emfs:" and "file:" protocols are not allowed for the requirement (${t})`)})}class G{constructor(){A(this,"_buffer",[]),A(this,"_promise"),A(this,"_resolve"),this._resolve=null,this._promise=null,this._notifyAll()}async _wait(){await this._promise}_notifyAll(){this._resolve&&this._resolve(),this._promise=new Promise(t=>this._resolve=t)}async dequeue(){for(;this._buffer.length===0;)await this._wait();return this._buffer.shift()}enqueue(t){this._buffer.push(t),this._notifyAll()}}function Y(e,t,o){const s=new G;o.addEventListener("message",r=>{s.enqueue(r.data)}),o.start();async function n(){return await s.dequeue()}async function i(r){const a=Object.fromEntries(r.toJs());o.postMessage(a)}return e(t,n,i)}const O="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";function W(e){return Array.from(Array(e)).map(()=>O[Math.floor(Math.random()*O.length)]).join("")}const z=`import ast
import os
import sys
import tokenize
import types
from inspect import CO_COROUTINE

from gradio.wasm_utils import app_id_context

# BSD 3-Clause License
#
# - Copyright (c) 2008-Present, IPython Development Team
# - Copyright (c) 2001-2007, Fernando Perez <fernando.perez@colorado.edu>
# - Copyright (c) 2001, Janko Hauser <jhauser@zscout.de>
# - Copyright (c) 2001, Nathaniel Gray <n8gray@caltech.edu>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Code modified from IPython (BSD license)
# Source: https://github.com/ipython/ipython/blob/master/IPython/utils/syspathcontext.py#L42
class modified_sys_path:  # noqa: N801
    """A context for prepending a directory to sys.path for a second."""

    def __init__(self, script_path: str):
        self._script_path = script_path
        self._added_path = False

    def __enter__(self):
        if self._script_path not in sys.path:
            sys.path.insert(0, self._script_path)
            self._added_path = True

    def __exit__(self, type, value, traceback):
        if self._added_path:
            try:
                sys.path.remove(self._script_path)
            except ValueError:
                # It's already removed.
                pass

        # Returning False causes any exceptions to be re-raised.
        return False


# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
# Copyright (c) Yuichiro Tachibana (2023)
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
def _new_module(name: str) -> types.ModuleType:
    """Create a new module with the given name."""
    return types.ModuleType(name)


def set_home_dir(home_dir: str) -> None:
    os.environ["HOME"] = home_dir
    os.chdir(home_dir)


async def _run_script(app_id: str, home_dir: str, script_path: str) -> None:
    # This function is based on the following code from Streamlit:
    # https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/runtime/scriptrunner/script_runner.py#L519-L554
    # with modifications to support top-level await.
    set_home_dir(home_dir)

    with tokenize.open(script_path) as f:
        filebody = f.read()

    await _run_code(app_id, home_dir, filebody, script_path)


async def _run_code(
        app_id: str,
        home_dir: str,
        filebody: str,
        script_path: str = '<string>'  # This default value follows the convention. Ref: https://docs.python.org/3/library/functions.html#compile
    ) -> None:
    set_home_dir(home_dir)

    # NOTE: In Streamlit, the bytecode caching mechanism has been introduced.
    # However, we skipped it here for simplicity and because Gradio doesn't need to rerun the script so frequently,
    # while we may do it in the future.
    bytecode = compile(  # type: ignore
        filebody,
        # Pass in the file path so it can show up in exceptions.
        script_path,
        # We're compiling entire blocks of Python, so we need "exec"
        # mode (as opposed to "eval" or "single").
        mode="exec",
        # Don't inherit any flags or "future" statements.
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, # Allow top-level await. Ref: https://github.com/whitphx/streamlit/commit/277dc580efb315a3e9296c9a0078c602a0904384
        dont_inherit=1,
        # Use the default optimization options.
        optimize=-1,
    )

    module = _new_module("__main__")

    # Install the fake module as the __main__ module. This allows
    # the pickle module to work inside the user's code, since it now
    # can know the module where the pickled objects stem from.
    # IMPORTANT: This means we can't use "if __name__ == '__main__'" in
    # our code, as it will point to the wrong module!!!
    sys.modules["__main__"] = module

    # Add special variables to the module's globals dict.
    module.__dict__["__file__"] = script_path

    with modified_sys_path(script_path), modified_sys_path(home_dir), app_id_context(app_id):
        # Allow top-level await. Ref: https://github.com/whitphx/streamlit/commit/277dc580efb315a3e9296c9a0078c602a0904384
        if bytecode.co_flags & CO_COROUTINE:
            # The source code includes top-level awaits, so the compiled code object is a coroutine.
            await eval(bytecode, module.__dict__)
        else:
            exec(bytecode, module.__dict__)
`,q=`# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
# Copyright (c) Yuichiro Tachibana (2023)
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

import fnmatch
import logging
import os
import sys
import types
from typing import Optional, Set

LOGGER = logging.getLogger(__name__)

#
# Copied from https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/file_util.py
#

def file_is_in_folder_glob(filepath, folderpath_glob) -> bool:
    """Test whether a file is in some folder with globbing support.

    Parameters
    ----------
    filepath : str
        A file path.
    folderpath_glob: str
        A path to a folder that may include globbing.

    """
    # Make the glob always end with "/*" so we match files inside subfolders of
    # folderpath_glob.
    if not folderpath_glob.endswith("*"):
        if folderpath_glob.endswith("/"):
            folderpath_glob += "*"
        else:
            folderpath_glob += "/*"

    file_dir = os.path.dirname(filepath) + "/"
    return fnmatch.fnmatch(file_dir, folderpath_glob)


def get_directory_size(directory: str) -> int:
    """Return the size of a directory in bytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def file_in_pythonpath(filepath) -> bool:
    """Test whether a filepath is in the same folder of a path specified in the PYTHONPATH env variable.


    Parameters
    ----------
    filepath : str
        An absolute file path.

    Returns
    -------
    boolean
        True if contained in PYTHONPATH, False otherwise. False if PYTHONPATH is not defined or empty.

    """
    pythonpath = os.environ.get("PYTHONPATH", "")
    if len(pythonpath) == 0:
        return False

    absolute_paths = [os.path.abspath(path) for path in pythonpath.split(os.pathsep)]
    return any(
        file_is_in_folder_glob(os.path.normpath(filepath), path)
        for path in absolute_paths
    )

#
# Copied from https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/watcher/local_sources_watcher.py
#

def get_module_paths(module: types.ModuleType) -> Set[str]:
    paths_extractors = [
        # https://docs.python.org/3/reference/datamodel.html
        # __file__ is the pathname of the file from which the module was loaded
        # if it was loaded from a file.
        # The __file__ attribute may be missing for certain types of modules
        lambda m: [m.__file__],
        # https://docs.python.org/3/reference/import.html#__spec__
        # The __spec__ attribute is set to the module spec that was used
        # when importing the module. one exception is __main__,
        # where __spec__ is set to None in some cases.
        # https://www.python.org/dev/peps/pep-0451/#id16
        # "origin" in an import context means the system
        # (or resource within a system) from which a module originates
        # ... It is up to the loader to decide on how to interpret
        # and use a module's origin, if at all.
        lambda m: [m.__spec__.origin],
        # https://www.python.org/dev/peps/pep-0420/
        # Handling of "namespace packages" in which the __path__ attribute
        # is a _NamespacePath object with a _path attribute containing
        # the various paths of the package.
        lambda m: list(m.__path__._path),
    ]

    all_paths = set()
    for extract_paths in paths_extractors:
        potential_paths = []
        try:
            potential_paths = extract_paths(module)
        except AttributeError:
            # Some modules might not have __file__ or __spec__ attributes.
            pass
        except Exception as e:
            LOGGER.warning(f"Examining the path of {module.__name__} raised: {e}")

        all_paths.update(
            [os.path.abspath(str(p)) for p in potential_paths if _is_valid_path(p)]
        )
    return all_paths


def _is_valid_path(path: Optional[str]) -> bool:
    return isinstance(path, str) and (os.path.isfile(path) or os.path.isdir(path))


#
# Original code
#

def unload_local_modules(target_dir_path: str = "."):
    """ Unload all modules that are in the target directory or in a subdirectory of it.
    It is necessary to unload modules before re-executing a script that imports the modules,
    so that the new version of the modules is loaded.
    The module unloading feature is extracted from Streamlit's LocalSourcesWatcher (https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/watcher/local_sources_watcher.py)
    and packaged as a standalone function.
    """
    target_dir_path = os.path.abspath(target_dir_path)
    loaded_modules = {} # filepath -> module_name

    # Copied from \`LocalSourcesWatcher.update_watched_modules()\`
    module_paths = {
        name: get_module_paths(module)
        for name, module in dict(sys.modules).items()
    }

    # Copied from \`LocalSourcesWatcher._register_necessary_watchers()\`
    for name, paths in module_paths.items():
        for path in paths:
            if file_is_in_folder_glob(path, target_dir_path) or file_in_pythonpath(path):
                loaded_modules[path] = name

    # Copied from \`LocalSourcesWatcher.on_file_changed()\`
    for module_name in loaded_modules.values():
        if module_name is not None and module_name in sys.modules:
            del sys.modules[module_name]
`;importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.1/full/pyodide.js");let p,w,N,R,C,I;async function j(e,t){console.debug("Loading Pyodide."),t("Loading Pyodide"),p=await loadPyodide({stdout:console.debug,stderr:console.error}),console.debug("Pyodide is loaded."),console.debug("Loading micropip"),t("Loading micropip"),await p.loadPackage("micropip"),w=p.pyimport("micropip"),console.debug("micropip is loaded.");const o=[e.gradioWheelUrl,e.gradioClientWheelUrl];console.debug("Loading Gradio wheels.",o),t("Loading Gradio wheels"),await p.loadPackage(["ssl","setuptools"]),await w.add_mock_package("ffmpy","0.3.0"),await w.install.callKwargs(o,{keep_going:!0}),console.debug("Gradio wheels are loaded."),console.debug("Mocking os module methods."),t("Mock os module methods"),await p.runPythonAsync(`
import os

os.link = lambda src, dst: None
`),console.debug("os module methods are mocked."),console.debug("Importing gradio package."),t("Importing gradio package"),await p.runPythonAsync("import gradio"),console.debug("gradio package is imported."),console.debug("Defining a ASGI wrapper function."),t("Defining a ASGI wrapper function"),await p.runPythonAsync(`
# Based on Shiny's App.call_pyodide().
# https://github.com/rstudio/py-shiny/blob/v0.3.3/shiny/_app.py#L224-L258
async def _call_asgi_app_from_js(app_id, scope, receive, send):
	# TODO: Pretty sure there are objects that need to be destroy()'d here?
	scope = scope.to_py()

	# ASGI requires some values to be byte strings, not character strings. Those are
	# not that easy to create in JavaScript, so we let the JS side pass us strings
	# and we convert them to bytes here.
	if "headers" in scope:
			# JS doesn't have \`bytes\` so we pass as strings and convert here
			scope["headers"] = [
					[value.encode("latin-1") for value in header]
					for header in scope["headers"]
			]
	if "query_string" in scope and scope["query_string"]:
			scope["query_string"] = scope["query_string"].encode("latin-1")
	if "raw_path" in scope and scope["raw_path"]:
			scope["raw_path"] = scope["raw_path"].encode("latin-1")

	async def rcv():
			event = await receive()
			py_event = event.to_py()
			if "body" in py_event:
					if isinstance(py_event["body"], memoryview):
							py_event["body"] = py_event["body"].tobytes()
			return py_event

	async def snd(event):
			await send(event)

	app = gradio.wasm_utils.get_registered_app(app_id)
	if app is None:
		raise RuntimeError("Gradio app has not been launched.")

	await app(scope, rcv, snd)
`),N=p.globals.get("_call_asgi_app_from_js"),console.debug("The ASGI wrapper function is defined."),console.debug("Mocking async libraries."),t("Mocking async libraries"),await p.runPythonAsync(`
async def mocked_anyio_to_thread_run_sync(func, *args, cancellable=False, limiter=None):
	return func(*args)

import anyio.to_thread
anyio.to_thread.run_sync = mocked_anyio_to_thread_run_sync
	`),console.debug("Async libraries are mocked."),console.debug("Setting up Python utility functions."),t("Setting up Python utility functions"),await p.runPythonAsync(z),R=p.globals.get("_run_code"),C=p.globals.get("_run_script"),await p.runPythonAsync(q),I=p.globals.get("unload_local_modules"),console.debug("Python utility functions are set up."),t("Initialization completed")}async function B(e,t,o){const s=b(e);console.debug("Creating a home directory for the app.",{appId:e,appHomeDir:s}),p.FS.mkdir(s),console.debug("Mounting files.",t.files),o("Mounting files"),await Promise.all(Object.keys(t.files).map(async n=>{const i=t.files[n];let r;"url"in i?(console.debug(`Fetch a file from ${i.url}`),r=await fetch(i.url).then(d=>d.arrayBuffer()).then(d=>new Uint8Array(d))):r=i.data;const{opts:a}=t.files[n],l=m(e,n);console.debug(`Write a file "${l}"`),E(p,l,r,a)})),console.debug("Files are mounted."),console.debug("Installing packages.",t.requirements),o("Installing packages"),await w.install.callKwargs(t.requirements,{keep_going:!0}),console.debug("Packages are installed."),t.requirements.includes("matplotlib")&&(console.debug("Setting matplotlib backend."),o("Setting matplotlib backend"),await p.runPythonAsync(`
try:
	import matplotlib
	matplotlib.use("agg")
except ImportError:
	pass
`),console.debug("matplotlib backend is set.")),o("App is now loaded")}const v=self;"postMessage"in v?L(v):v.onconnect=e=>{const t=e.ports[0];L(t),t.start()};let _;function L(e){const t=W(8);console.debug("Set up a new app.",{appId:t});const o=n=>{const i={type:"progress-update",data:{log:n}};e.postMessage(i)};let s;e.onmessage=async function(n){const i=n.data;console.debug("worker.onmessage",i);const r=n.ports[0];try{if(i.type==="init-env"){_==null?_=j(i.data,o):o("Pyodide environment initialization is ongoing in another session"),_.then(()=>{const a={type:"reply:success",data:null};r.postMessage(a)}).catch(a=>{const l={type:"reply:error",error:a};r.postMessage(l)});return}if(_==null)throw new Error("Pyodide Initialization is not started.");if(await _,i.type==="init-app"){s=B(t,i.data,o);const a={type:"reply:success",data:null};r.postMessage(a);return}if(s==null)throw new Error("App initialization is not started.");switch(await s,i.type){case"echo":{const a={type:"reply:success",data:i.data};r.postMessage(a);break}case"run-python-code":{I(),await R(t,b(t),i.data.code);const a={type:"reply:success",data:null};r.postMessage(a);break}case"run-python-file":{I(),await C(t,b(t),i.data.path);const a={type:"reply:success",data:null};r.postMessage(a);break}case"asgi-request":{console.debug("ASGI request",i.data),Y(N.bind(null,t),i.data.scope,r);break}case"file:write":{const{path:a,data:l,opts:d}=i.data,c=m(t,a);console.debug(`Write a file "${c}"`),E(p,c,l,d);const f={type:"reply:success",data:null};r.postMessage(f);break}case"file:rename":{const{oldPath:a,newPath:l}=i.data,d=m(t,a),c=m(t,l);console.debug(`Rename "${d}" to ${c}`),H(p,d,c);const f={type:"reply:success",data:null};r.postMessage(f);break}case"file:unlink":{const{path:a}=i.data,l=m(t,a);console.debug(`Remove "${l}`),p.FS.unlink(l);const d={type:"reply:success",data:null};r.postMessage(d);break}case"install":{const{requirements:a}=i.data,l=p.pyimport("micropip");console.debug("Install the requirements:",a),U(a),await l.install.callKwargs(a,{keep_going:!0}).then(()=>{if(a.includes("matplotlib"))return p.runPythonAsync(`
try:
	import matplotlib
	matplotlib.use("agg")
except ImportError:
	pass
`)}).then(()=>{console.debug("Successfully installed");const d={type:"reply:success",data:null};r.postMessage(d)});break}}}catch(a){if(console.error(a),!(a instanceof Error))throw a;const l=new Error(a.message);l.name=a.name,l.stack=a.stack;const d={type:"reply:error",error:l};r.postMessage(d)}}}})();
//# sourceMappingURL=webworker-CDafrY5U.js.map
