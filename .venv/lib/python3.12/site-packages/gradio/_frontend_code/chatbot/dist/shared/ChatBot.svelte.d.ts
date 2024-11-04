import { SvelteComponent } from "svelte";
import { type UndoRetryData } from "./utils";
import type { NormalisedMessage } from "../types";
import type { SelectData, LikeData } from "@gradio/utils";
import type { ExampleMessage } from "../types";
import type { FileData, Client } from "@gradio/client";
import type { I18nFormatter } from "js/core/src/gradio_helper";
import { Gradio } from "@gradio/utils";
declare const __propDef: {
    props: {
        value?: (NormalisedMessage[] | null) | undefined;
        _fetch: typeof fetch;
        load_component: Gradio["load_component"];
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        pending_message?: boolean | undefined;
        generating?: boolean | undefined;
        selectable?: boolean | undefined;
        likeable?: boolean | undefined;
        show_share_button?: boolean | undefined;
        show_copy_all_button?: boolean | undefined;
        rtl?: boolean | undefined;
        show_copy_button?: boolean | undefined;
        avatar_images?: [FileData | null, FileData | null] | undefined;
        sanitize_html?: boolean | undefined;
        bubble_full_width?: boolean | undefined;
        render_markdown?: boolean | undefined;
        line_breaks?: boolean | undefined;
        autoscroll?: boolean | undefined;
        theme_mode: "system" | "light" | "dark";
        i18n: I18nFormatter;
        layout?: ("bubble" | "panel") | undefined;
        placeholder?: (string | null) | undefined;
        upload: Client["upload"];
        msg_format?: ("tuples" | "messages") | undefined;
        examples?: (ExampleMessage[] | null) | undefined;
        _retryable?: boolean | undefined;
        _undoable?: boolean | undefined;
        like_user_message?: boolean | undefined;
        root: string;
    };
    events: {
        change: CustomEvent<undefined>;
        select: CustomEvent<SelectData>;
        like: CustomEvent<LikeData>;
        undo: CustomEvent<UndoRetryData>;
        retry: CustomEvent<UndoRetryData>;
        clear: CustomEvent<undefined>;
        share: CustomEvent<any>;
        error: CustomEvent<string>;
        example_select: CustomEvent<SelectData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ChatBotProps = typeof __propDef.props;
export type ChatBotEvents = typeof __propDef.events;
export type ChatBotSlots = typeof __propDef.slots;
export default class ChatBot extends SvelteComponent<ChatBotProps, ChatBotEvents, ChatBotSlots> {
}
export {};
