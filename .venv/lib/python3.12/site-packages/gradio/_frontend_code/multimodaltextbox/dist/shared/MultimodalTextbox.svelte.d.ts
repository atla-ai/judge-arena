import { SvelteComponent } from "svelte";
import type { FileData, Client } from "@gradio/client";
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value?: {
            text: string;
            files: FileData[];
        } | undefined;
        value_is_output?: boolean | undefined;
        lines?: number | undefined;
        placeholder?: string | undefined;
        disabled?: boolean | undefined;
        label: string;
        info?: string | undefined;
        show_label?: boolean | undefined;
        container?: boolean | undefined;
        max_lines: number;
        submit_btn?: (string | boolean | null) | undefined;
        stop_btn?: (string | boolean | null) | undefined;
        rtl?: boolean | undefined;
        autofocus?: boolean | undefined;
        text_align?: "left" | "right" | undefined;
        autoscroll?: boolean | undefined;
        root: string;
        file_types?: (string[] | null) | undefined;
        max_file_size?: (number | null) | undefined;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        file_count?: ("single" | "multiple" | "directory") | undefined;
        dragging?: boolean | undefined;
    };
    events: {
        dragover: DragEvent;
        error: CustomEvent<any>;
        blur: CustomEvent<any>;
        focus: CustomEvent<any>;
        change: CustomEvent<{
            text: string;
            files: FileData[];
        }>;
        submit: CustomEvent<undefined>;
        stop: CustomEvent<undefined>;
        select: CustomEvent<SelectData>;
        input: CustomEvent<undefined>;
        drag: CustomEvent<boolean>;
        upload: CustomEvent<any>;
        clear: CustomEvent<undefined>;
        load: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type MultimodalTextboxProps = typeof __propDef.props;
export type MultimodalTextboxEvents = typeof __propDef.events;
export type MultimodalTextboxSlots = typeof __propDef.slots;
export default class MultimodalTextbox extends SvelteComponent<MultimodalTextboxProps, MultimodalTextboxEvents, MultimodalTextboxSlots> {
}
export {};
