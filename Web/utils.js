//utils.js
console.log("utils.js loaded");
import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

// Create a read-only string widget
export function createTextWidget(app, node, widgetName, styles = {}) {
    const widget = ComfyWidgets["STRING"](node, widgetName, ["STRING", {multiline: true}], app).widget;
    widget.inputEl.readOnly = true;
    Object.assign(widget.inputEl.style, styles);
    return widget;
}


// Create a custom HTML widget using addDOMWidget for proper integration
export function createHTMLWidget(node, widgetName, styles = {}) {
    console.log("Creating HTML widget container...");
    const container = document.createElement('div');
    container.style.width = '100%';
    container.style.height = 'auto';
    container.style.overflowY = 'auto';

    Object.assign(container.style, styles);

    return {
        name: widgetName,
        type: 'html',
        widget: container,
        onResize: function (width, height) {
            container.style.width = width + 'px';
            container.style.height = height + 'px';
            console.log("Resizing HTML widget:", width, height);
        },
    };
}