<script>import { createEventDispatcher, tick } from "svelte";
import { Upload, ModifyUpload } from "@gradio/upload";
import { BlockLabel } from "@gradio/atoms";
import { File } from "@gradio/icons";
import FilePreview from "./FilePreview.svelte";
export let value;
export let label;
export let show_label = true;
export let file_count = "single";
export let file_types = null;
export let selectable = false;
export let root;
export let height = void 0;
export let i18n;
export let max_file_size = null;
export let upload;
export let stream_handler;
export let uploading = false;
async function handle_upload({
  detail
}) {
  value = detail;
  await tick();
  dispatch("change", value);
  dispatch("upload", detail);
}
function handle_clear() {
  value = null;
  dispatch("change", null);
  dispatch("clear");
}
const dispatch = createEventDispatcher();
let dragging = false;
$:
  dispatch("drag", dragging);
</script>

<BlockLabel {show_label} Icon={File} float={!value} label={label || "File"} />

{#if value && (Array.isArray(value) ? value.length > 0 : true)}
	<ModifyUpload {i18n} on:clear={handle_clear} />
	<FilePreview
		{i18n}
		on:select
		{selectable}
		{value}
		{height}
		on:change
		on:delete
	/>
{:else}
	<Upload
		on:load={handle_upload}
		filetype={file_types}
		{file_count}
		{max_file_size}
		{root}
		bind:dragging
		bind:uploading
		on:error
		{stream_handler}
		{upload}
	>
		<slot />
	</Upload>
{/if}
