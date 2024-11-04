import { SvelteComponent } from "svelte";
import type { FileData, Client } from "@gradio/client";
import type { NormalisedMessage } from "../types";
import type { I18nFormatter } from "js/core/src/gradio_helper";
import type { ComponentType } from "svelte";
declare const __propDef: {
    props: {
        value: NormalisedMessage[];
        avatar_img: FileData | null;
        opposite_avatar_img?: FileData | null;
        role?: string | undefined;
        messages?: NormalisedMessage[] | undefined;
        layout: "bubble" | "panel";
        bubble_full_width: boolean;
        render_markdown: boolean;
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        sanitize_html: boolean;
        selectable: boolean;
        _fetch: typeof fetch;
        rtl: boolean;
        dispatch: any;
        i18n: I18nFormatter;
        line_breaks: boolean;
        upload: Client["upload"];
        target: HTMLElement | null;
        root: string;
        theme_mode: "light" | "dark" | "system";
        _components: Record<string, ComponentType<SvelteComponent>>;
        i: number;
        show_copy_button: boolean;
        generating: boolean;
        show_like: boolean;
        show_retry: boolean;
        show_undo: boolean;
        msg_format: "tuples" | "messages";
        handle_action: (selected: string | null) => void;
        scroll: () => void;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type MessageProps = typeof __propDef.props;
export type MessageEvents = typeof __propDef.events;
export type MessageSlots = typeof __propDef.slots;
export default class Message extends SvelteComponent<MessageProps, MessageEvents, MessageSlots> {
}
export {};
