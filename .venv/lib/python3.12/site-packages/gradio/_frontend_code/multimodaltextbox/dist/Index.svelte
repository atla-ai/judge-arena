<svelte:options accessors={true} />

<script context="module">export { default as BaseMultimodalTextbox } from "./shared/MultimodalTextbox.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import MultimodalTextbox from "./shared/MultimodalTextbox.svelte";
import { Block } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let gradio;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = {
  text: "",
  files: []
};
export let file_types = null;
export let lines;
export let placeholder = "";
export let label = "MultimodalTextbox";
export let info = void 0;
export let show_label;
export let max_lines;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let submit_btn = null;
export let stop_btn = null;
export let loading_status = void 0;
export let value_is_output = false;
export let rtl = false;
export let text_align = void 0;
export let autofocus = false;
export let autoscroll = true;
export let interactive;
export let root;
export let file_count;
let dragging;
</script>

<Block
	{visible}
	{elem_id}
	elem_classes={[...elem_classes, "multimodal-textbox"]}
	{scale}
	{min_width}
	allow_overflow={false}
	padding={container}
	border_mode={dragging ? "focus" : "base"}
>
	{#if loading_status}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
	{/if}

	<MultimodalTextbox
		bind:value
		bind:value_is_output
		bind:dragging
		{file_types}
		{root}
		{label}
		{info}
		{show_label}
		{lines}
		{rtl}
		{text_align}
		max_lines={!max_lines ? lines + 1 : max_lines}
		{placeholder}
		{submit_btn}
		{stop_btn}
		{autofocus}
		{container}
		{autoscroll}
		{file_count}
		max_file_size={gradio.max_file_size}
		on:change={() => gradio.dispatch("change", value)}
		on:input={() => gradio.dispatch("input")}
		on:submit={() => gradio.dispatch("submit")}
		on:stop={() => gradio.dispatch("stop")}
		on:blur={() => gradio.dispatch("blur")}
		on:select={(e) => gradio.dispatch("select", e.detail)}
		on:focus={() => gradio.dispatch("focus")}
		on:error={({ detail }) => {
			gradio.dispatch("error", detail);
		}}
		disabled={!interactive}
		upload={(...args) => gradio.client.upload(...args)}
		stream_handler={(...args) => gradio.client.stream(...args)}
	/>
</Block>
