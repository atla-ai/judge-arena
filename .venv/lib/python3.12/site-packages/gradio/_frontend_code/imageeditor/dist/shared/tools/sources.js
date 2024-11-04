import { Sprite, Texture, DisplayObject, Color, RenderTexture } from "pixi.js";
import {} from "../utils/commands";
import { make_graphics } from "../utils/pixi";
/**
 * Adds a background image to the canvas.
 * @param container The container to add the image to.
 * @param renderer The renderer to use for the image.
 * @param background The background image to add.
 * @param resize The function to resize the canvas.
 * @returns A command that can be used to undo the action.
 */
export function add_bg_image(container, renderer, background, resize) {
    let sprite;
    return {
        async start() {
            container.removeChildren();
            const img = await createImageBitmap(background);
            const bitmap_texture = Texture.from(img);
            sprite = new Sprite(bitmap_texture);
            return [sprite.width, sprite.height];
        },
        async execute() {
            // renderer.resize(sprite.width, sprite.height);
            resize(sprite.width, sprite.height);
            sprite.zIndex = 0;
            container.addChild(sprite);
        },
        undo() {
            container.removeChildAt(0);
        }
    };
}
/**
 * Adds a background color to the canvas.
 * @param container The container to add the image to.
 * @param renderer The renderer to use for the image.
 * @param color The background color to add.
 * @param width The width of the background.
 * @param height The height of the background.
 * @param resize The function to resize the canvas.
 * @returns A command that can be used to undo the action.
 */
export function add_bg_color(container, renderer, color, width, height, resize) {
    let sprite;
    return {
        start() {
            container.removeChildren();
            const graphics = make_graphics(1);
            const texture = RenderTexture.create({
                width,
                height
            });
            graphics.beginFill(new Color(color));
            graphics.drawRect(0, 0, width, height);
            graphics.endFill();
            renderer.render(graphics, { renderTexture: texture });
            sprite = new Sprite(texture);
            return [sprite.width, sprite.height];
        },
        async execute() {
            resize(sprite.width, sprite.height);
            sprite.zIndex = 1;
            container.addChild(sprite);
        },
        undo() {
            container.removeChildren();
        }
    };
}
