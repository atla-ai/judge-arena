import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        layout?: string | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type PendingProps = typeof __propDef.props;
export type PendingEvents = typeof __propDef.events;
export type PendingSlots = typeof __propDef.slots;
export default class Pending extends SvelteComponent<PendingProps, PendingEvents, PendingSlots> {
}
export {};
