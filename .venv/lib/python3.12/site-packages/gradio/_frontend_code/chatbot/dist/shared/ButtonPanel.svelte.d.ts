import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
import type { NormalisedMessage } from "../types";
declare const __propDef: {
    props: {
        likeable: boolean;
        show_retry: boolean;
        show_undo: boolean;
        show_copy_button: boolean;
        show: boolean;
        message: NormalisedMessage | NormalisedMessage[];
        position: "right" | "left";
        avatar: FileData | null;
        generating: boolean;
        handle_action: (selected: string | null) => void;
        layout: "bubble" | "panel";
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ButtonPanelProps = typeof __propDef.props;
export type ButtonPanelEvents = typeof __propDef.events;
export type ButtonPanelSlots = typeof __propDef.slots;
export default class ButtonPanel extends SvelteComponent<ButtonPanelProps, ButtonPanelEvents, ButtonPanelSlots> {
}
export {};
