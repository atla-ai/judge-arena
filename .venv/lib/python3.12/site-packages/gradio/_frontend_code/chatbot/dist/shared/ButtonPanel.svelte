<script>import LikeDislike from "./LikeDislike.svelte";
import Copy from "./Copy.svelte";
import DownloadIcon from "./Download.svelte";
import { DownloadLink } from "@gradio/wasm/svelte";
import { is_component_message } from "./utils";
import { Retry, Undo } from "@gradio/icons";
import { IconButtonWrapper, IconButton } from "@gradio/atoms";
export let likeable;
export let show_retry;
export let show_undo;
export let show_copy_button;
export let show;
export let message;
export let position;
export let avatar;
export let generating;
export let handle_action;
export let layout;
function is_all_text(message2) {
  return Array.isArray(message2) && message2.every((m) => typeof m.content === "string") || !Array.isArray(message2) && typeof message2.content === "string";
}
function all_text(message2) {
  if (Array.isArray(message2)) {
    return message2.map((m) => m.content).join("\n");
  }
  return message2.content;
}
$:
  message_text = is_all_text(message) ? all_text(message) : "";
$:
  show_copy = show_copy_button && message && is_all_text(message);
$:
  show_download = !Array.isArray(message) && is_component_message(message) && message.content.value?.url;
</script>

{#if show}
	<div
		class="message-buttons-{position} {layout} message-buttons {avatar !==
			null && 'with-avatar'}"
	>
		<IconButtonWrapper top_panel={false}>
			{#if show_copy}
				<Copy value={message_text} />
			{/if}
			{#if show_download && !Array.isArray(message) && is_component_message(message)}
				<DownloadLink
					href={message?.content?.value.url}
					download={message.content.value.orig_name || "image"}
				>
					<IconButton Icon={DownloadIcon} />
				</DownloadLink>
			{/if}
			{#if show_retry}
				<IconButton
					Icon={Retry}
					label="Retry"
					on:click={() => handle_action("retry")}
					disabled={generating}
				/>
			{/if}
			{#if show_undo}
				<IconButton
					label="Undo"
					Icon={Undo}
					on:click={() => handle_action("undo")}
					disabled={generating}
				/>
			{/if}
			{#if likeable}
				<LikeDislike {handle_action} />
			{/if}
		</IconButtonWrapper>
	</div>
{/if}

<style>
	.bubble :global(.icon-button-wrapper) {
		margin: 0px calc(var(--spacing-xl) * 2);
	}

	.message-buttons-left {
		align-self: flex-start;
	}

	.bubble.message-buttons-right {
		align-self: flex-end;
	}

	.message-buttons-right :global(.icon-button-wrapper) {
		margin-left: auto;
	}

	.bubble.with-avatar {
		margin-left: calc(var(--spacing-xl) * 5);
		margin-right: calc(var(--spacing-xl) * 5);
	}

	.panel {
		display: flex;
		align-self: flex-start;
		padding: 0 var(--spacing-xl);
		z-index: var(--layer-1);
	}
</style>
