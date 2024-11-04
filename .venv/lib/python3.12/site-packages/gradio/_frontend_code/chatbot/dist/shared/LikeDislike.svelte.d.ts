import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        handle_action: (selected: string | null) => void;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type LikeDislikeProps = typeof __propDef.props;
export type LikeDislikeEvents = typeof __propDef.events;
export type LikeDislikeSlots = typeof __propDef.slots;
export default class LikeDislike extends SvelteComponent<LikeDislikeProps, LikeDislikeEvents, LikeDislikeSlots> {
}
export {};
