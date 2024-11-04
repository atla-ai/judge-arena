import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
import type { I18nFormatter, SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: FileData | FileData[];
        selectable?: boolean | undefined;
        height?: number | undefined;
        i18n: I18nFormatter;
    };
    events: {
        select: CustomEvent<SelectData>;
        change: CustomEvent<any>;
        delete: CustomEvent<FileData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type FilePreviewProps = typeof __propDef.props;
export type FilePreviewEvents = typeof __propDef.events;
export type FilePreviewSlots = typeof __propDef.slots;
export default class FilePreview extends SvelteComponent<FilePreviewProps, FilePreviewEvents, FilePreviewSlots> {
}
export {};
