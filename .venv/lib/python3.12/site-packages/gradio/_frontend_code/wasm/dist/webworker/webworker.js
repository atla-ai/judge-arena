var M=Object.defineProperty;var D=(o,e,t)=>e in o?M(o,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):o[e]=t;var y=(o,e,t)=>(D(o,typeof e!="symbol"?e+"":e,t),t);function F(o){return o&&o.__esModule&&Object.prototype.hasOwnProperty.call(o,"default")?o.default:o}function u(o){if(typeof o!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(o))}function T(o,e){for(var t="",r=0,s=-1,a=0,n,i=0;i<=o.length;++i){if(i<o.length)n=o.charCodeAt(i);else{if(n===47)break;n=47}if(n===47){if(!(s===i-1||a===1))if(s!==i-1&&a===2){if(t.length<2||r!==2||t.charCodeAt(t.length-1)!==46||t.charCodeAt(t.length-2)!==46){if(t.length>2){var l=t.lastIndexOf("/");if(l!==t.length-1){l===-1?(t="",r=0):(t=t.slice(0,l),r=t.length-1-t.lastIndexOf("/")),s=i,a=0;continue}}else if(t.length===2||t.length===1){t="",r=0,s=i,a=0;continue}}e&&(t.length>0?t+="/..":t="..",r=2)}else t.length>0?t+="/"+o.slice(s+1,i):t=o.slice(s+1,i),r=i-s-1;s=i,a=0}else n===46&&a!==-1?++a:a=-1}return t}function H(o,e){var t=e.dir||e.root,r=e.base||(e.name||"")+(e.ext||"");return t?t===e.root?t+r:t+o+r:r}var f={resolve:function(){for(var e="",t=!1,r,s=arguments.length-1;s>=-1&&!t;s--){var a;s>=0?a=arguments[s]:(r===void 0&&(r=process.cwd()),a=r),u(a),a.length!==0&&(e=a+"/"+e,t=a.charCodeAt(0)===47)}return e=T(e,!t),t?e.length>0?"/"+e:"/":e.length>0?e:"."},normalize:function(e){if(u(e),e.length===0)return".";var t=e.charCodeAt(0)===47,r=e.charCodeAt(e.length-1)===47;return e=T(e,!t),e.length===0&&!t&&(e="."),e.length>0&&r&&(e+="/"),t?"/"+e:e},isAbsolute:function(e){return u(e),e.length>0&&e.charCodeAt(0)===47},join:function(){if(arguments.length===0)return".";for(var e,t=0;t<arguments.length;++t){var r=arguments[t];u(r),r.length>0&&(e===void 0?e=r:e+="/"+r)}return e===void 0?".":f.normalize(e)},relative:function(e,t){if(u(e),u(t),e===t||(e=f.resolve(e),t=f.resolve(t),e===t))return"";for(var r=1;r<e.length&&e.charCodeAt(r)===47;++r);for(var s=e.length,a=s-r,n=1;n<t.length&&t.charCodeAt(n)===47;++n);for(var i=t.length,l=i-n,d=a<l?a:l,h=-1,c=0;c<=d;++c){if(c===d){if(l>d){if(t.charCodeAt(n+c)===47)return t.slice(n+c+1);if(c===0)return t.slice(n+c)}else a>d&&(e.charCodeAt(r+c)===47?h=c:c===0&&(h=0));break}var I=e.charCodeAt(r+c),P=t.charCodeAt(n+c);if(I!==P)break;I===47&&(h=c)}var m="";for(c=r+h+1;c<=s;++c)(c===s||e.charCodeAt(c)===47)&&(m.length===0?m+="..":m+="/..");return m.length>0?m+t.slice(n+h):(n+=h,t.charCodeAt(n)===47&&++n,t.slice(n))},_makeLong:function(e){return e},dirname:function(e){if(u(e),e.length===0)return".";for(var t=e.charCodeAt(0),r=t===47,s=-1,a=!0,n=e.length-1;n>=1;--n)if(t=e.charCodeAt(n),t===47){if(!a){s=n;break}}else a=!1;return s===-1?r?"/":".":r&&s===1?"//":e.slice(0,s)},basename:function(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');u(e);var r=0,s=-1,a=!0,n;if(t!==void 0&&t.length>0&&t.length<=e.length){if(t.length===e.length&&t===e)return"";var i=t.length-1,l=-1;for(n=e.length-1;n>=0;--n){var d=e.charCodeAt(n);if(d===47){if(!a){r=n+1;break}}else l===-1&&(a=!1,l=n+1),i>=0&&(d===t.charCodeAt(i)?--i===-1&&(s=n):(i=-1,s=l))}return r===s?s=l:s===-1&&(s=e.length),e.slice(r,s)}else{for(n=e.length-1;n>=0;--n)if(e.charCodeAt(n)===47){if(!a){r=n+1;break}}else s===-1&&(a=!1,s=n+1);return s===-1?"":e.slice(r,s)}},extname:function(e){u(e);for(var t=-1,r=0,s=-1,a=!0,n=0,i=e.length-1;i>=0;--i){var l=e.charCodeAt(i);if(l===47){if(!a){r=i+1;break}continue}s===-1&&(a=!1,s=i+1),l===46?t===-1?t=i:n!==1&&(n=1):t!==-1&&(n=-1)}return t===-1||s===-1||n===0||n===1&&t===s-1&&t===r+1?"":e.slice(t,s)},format:function(e){if(e===null||typeof e!="object")throw new TypeError('The "pathObject" argument must be of type Object. Received type '+typeof e);return H("/",e)},parse:function(e){u(e);var t={root:"",dir:"",base:"",ext:"",name:""};if(e.length===0)return t;var r=e.charCodeAt(0),s=r===47,a;s?(t.root="/",a=1):a=0;for(var n=-1,i=0,l=-1,d=!0,h=e.length-1,c=0;h>=a;--h){if(r=e.charCodeAt(h),r===47){if(!d){i=h+1;break}continue}l===-1&&(d=!1,l=h+1),r===46?n===-1?n=h:c!==1&&(c=1):n!==-1&&(c=-1)}return n===-1||l===-1||c===0||c===1&&n===l-1&&n===i+1?l!==-1&&(i===0&&s?t.base=t.name=e.slice(1,l):t.base=t.name=e.slice(i,l)):(i===0&&s?(t.name=e.slice(1,n),t.base=e.slice(1,l)):(t.name=e.slice(i,n),t.base=e.slice(i,l)),t.ext=e.slice(n,l)),i>0?t.dir=e.slice(0,i-1):s&&(t.dir="/"),t},sep:"/",delimiter:":",win32:null,posix:null};f.posix=f;var U=f;const w=F(U),x="/home/pyodide",v=o=>`${x}/${o}`,g=(o,e)=>(w.normalize(e),w.resolve(v(o),e));function R(o,e){const t=w.normalize(e),s=w.dirname(t).split("/"),a=[];for(const n of s){a.push(n);const i=a.join("/");if(o.FS.analyzePath(i).exists){if(o.FS.isDir(i))throw new Error(`"${i}" already exists and is not a directory.`);continue}try{o.FS.mkdir(i)}catch(l){throw console.error(`Failed to create a directory "${i}"`),l}}}function N(o,e,t,r){R(o,e),o.FS.writeFile(e,t,r)}function z(o,e,t){R(o,t),o.FS.rename(e,t)}function G(o){o.forEach(e=>{let t;try{t=new URL(e)}catch{return}if(t.protocol==="emfs:"||t.protocol==="file:")throw new Error(`"emfs:" and "file:" protocols are not allowed for the requirement (${e})`)})}class W{constructor(){y(this,"_buffer",[]);y(this,"_promise");y(this,"_resolve");this._resolve=null,this._promise=null,this._notifyAll()}async _wait(){await this._promise}_notifyAll(){this._resolve&&this._resolve(),this._promise=new Promise(e=>this._resolve=e)}async dequeue(){for(;this._buffer.length===0;)await this._wait();return this._buffer.shift()}enqueue(e){this._buffer.push(e),this._notifyAll()}}function Y(o,e,t){const r=new W;t.addEventListener("message",n=>{r.enqueue(n.data)}),t.start();async function s(){return await r.dequeue()}async function a(n){const i=Object.fromEntries(n.toJs());t.postMessage(i)}return o(e,s,a)}const E="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";function q(o){return Array.from(Array(o)).map(()=>E[Math.floor(Math.random()*E.length)]).join("")}const B=`import ast
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
`,j=`# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
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
`;importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.1/full/pyodide.js");let p,b,L,C,k,S;async function V(o,e){console.debug("Loading Pyodide."),e("Loading Pyodide"),p=await loadPyodide({stdout:console.debug,stderr:console.error}),console.debug("Pyodide is loaded."),console.debug("Loading micropip"),e("Loading micropip"),await p.loadPackage("micropip"),b=p.pyimport("micropip"),console.debug("micropip is loaded.");const t=[o.gradioWheelUrl,o.gradioClientWheelUrl];console.debug("Loading Gradio wheels.",t),e("Loading Gradio wheels"),await p.loadPackage(["ssl","setuptools"]),await b.add_mock_package("ffmpy","0.3.0"),await b.install.callKwargs(t,{keep_going:!0}),console.debug("Gradio wheels are loaded."),console.debug("Mocking os module methods."),e("Mock os module methods"),await p.runPythonAsync(`
import os

os.link = lambda src, dst: None
`),console.debug("os module methods are mocked."),console.debug("Importing gradio package."),e("Importing gradio package"),await p.runPythonAsync("import gradio"),console.debug("gradio package is imported."),console.debug("Defining a ASGI wrapper function."),e("Defining a ASGI wrapper function"),await p.runPythonAsync(`
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
`),L=p.globals.get("_call_asgi_app_from_js"),console.debug("The ASGI wrapper function is defined."),console.debug("Mocking async libraries."),e("Mocking async libraries"),await p.runPythonAsync(`
async def mocked_anyio_to_thread_run_sync(func, *args, cancellable=False, limiter=None):
	return func(*args)

import anyio.to_thread
anyio.to_thread.run_sync = mocked_anyio_to_thread_run_sync
	`),console.debug("Async libraries are mocked."),console.debug("Setting up Python utility functions."),e("Setting up Python utility functions"),await p.runPythonAsync(B),C=p.globals.get("_run_code"),k=p.globals.get("_run_script"),await p.runPythonAsync(j),S=p.globals.get("unload_local_modules"),console.debug("Python utility functions are set up."),e("Initialization completed")}async function $(o,e,t){const r=v(o);console.debug("Creating a home directory for the app.",{appId:o,appHomeDir:r}),p.FS.mkdir(r),console.debug("Mounting files.",e.files),t("Mounting files"),await Promise.all(Object.keys(e.files).map(async s=>{const a=e.files[s];let n;"url"in a?(console.debug(`Fetch a file from ${a.url}`),n=await fetch(a.url).then(d=>d.arrayBuffer()).then(d=>new Uint8Array(d))):n=a.data;const{opts:i}=e.files[s],l=g(o,s);console.debug(`Write a file "${l}"`),N(p,l,n,i)})),console.debug("Files are mounted."),console.debug("Installing packages.",e.requirements),t("Installing packages"),await b.install.callKwargs(e.requirements,{keep_going:!0}),console.debug("Packages are installed."),e.requirements.includes("matplotlib")&&(console.debug("Setting matplotlib backend."),t("Setting matplotlib backend"),await p.runPythonAsync(`
try:
	import matplotlib
	matplotlib.use("agg")
except ImportError:
	pass
`),console.debug("matplotlib backend is set.")),t("App is now loaded")}const A=self;"postMessage"in A?O(A):A.onconnect=o=>{const e=o.ports[0];O(e),e.start()};let _;function O(o){const e=q(8);console.debug("Set up a new app.",{appId:e});const t=s=>{const a={type:"progress-update",data:{log:s}};o.postMessage(a)};let r;o.onmessage=async function(s){const a=s.data;console.debug("worker.onmessage",a);const n=s.ports[0];try{if(a.type==="init-env"){_==null?_=V(a.data,t):t("Pyodide environment initialization is ongoing in another session"),_.then(()=>{const i={type:"reply:success",data:null};n.postMessage(i)}).catch(i=>{const l={type:"reply:error",error:i};n.postMessage(l)});return}if(_==null)throw new Error("Pyodide Initialization is not started.");if(await _,a.type==="init-app"){r=$(e,a.data,t);const i={type:"reply:success",data:null};n.postMessage(i);return}if(r==null)throw new Error("App initialization is not started.");switch(await r,a.type){case"echo":{const i={type:"reply:success",data:a.data};n.postMessage(i);break}case"run-python-code":{S(),await C(e,v(e),a.data.code);const i={type:"reply:success",data:null};n.postMessage(i);break}case"run-python-file":{S(),await k(e,v(e),a.data.path);const i={type:"reply:success",data:null};n.postMessage(i);break}case"asgi-request":{console.debug("ASGI request",a.data),Y(L.bind(null,e),a.data.scope,n);break}case"file:write":{const{path:i,data:l,opts:d}=a.data,h=g(e,i);console.debug(`Write a file "${h}"`),N(p,h,l,d);const c={type:"reply:success",data:null};n.postMessage(c);break}case"file:rename":{const{oldPath:i,newPath:l}=a.data,d=g(e,i),h=g(e,l);console.debug(`Rename "${d}" to ${h}`),z(p,d,h);const c={type:"reply:success",data:null};n.postMessage(c);break}case"file:unlink":{const{path:i}=a.data,l=g(e,i);console.debug(`Remove "${l}`),p.FS.unlink(l);const d={type:"reply:success",data:null};n.postMessage(d);break}case"install":{const{requirements:i}=a.data,l=p.pyimport("micropip");console.debug("Install the requirements:",i),G(i),await l.install.callKwargs(i,{keep_going:!0}).then(()=>{if(i.includes("matplotlib"))return p.runPythonAsync(`
try:
	import matplotlib
	matplotlib.use("agg")
except ImportError:
	pass
`)}).then(()=>{console.debug("Successfully installed");const d={type:"reply:success",data:null};n.postMessage(d)});break}}}catch(i){if(console.error(i),!(i instanceof Error))throw i;const l=new Error(i.message);l.name=i.name,l.stack=i.stack;const d={type:"reply:error",error:l};n.postMessage(d)}}}
