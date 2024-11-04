<script>import { onDestroy } from "svelte";
import { Copy, Check } from "@gradio/icons";
import { IconButton } from "@gradio/atoms";
let copied = false;
export let value;
let timer;
function copy_feedback() {
  copied = true;
  if (timer)
    clearTimeout(timer);
  timer = setTimeout(() => {
    copied = false;
  }, 2e3);
}
async function handle_copy() {
  if ("clipboard" in navigator) {
    await navigator.clipboard.writeText(value);
    copy_feedback();
  } else {
    const textArea = document.createElement("textarea");
    textArea.value = value;
    textArea.style.position = "absolute";
    textArea.style.left = "-999999px";
    document.body.prepend(textArea);
    textArea.select();
    try {
      document.execCommand("copy");
      copy_feedback();
    } catch (error) {
      console.error(error);
    } finally {
      textArea.remove();
    }
  }
}
onDestroy(() => {
  if (timer)
    clearTimeout(timer);
});
</script>

<IconButton
	on:click={handle_copy}
	label={copied ? "Copied message" : "Copy message"}
	Icon={copied ? Check : Copy}
/>
