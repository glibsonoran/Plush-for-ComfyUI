// @ts-check
//This code is modified from pysssss bindings.js
//All credit and kudos to the original author
//--------------------------------
// These imports are from ComfyUI's core files
// The relative path "../../../../scripts/" goes up from:
// extension/web/js/binding.js -> to ComfyUI's main scripts directory
import { ComfyWidgets } from "../../../../scripts/widgets.js";
import { api } from "../../../../scripts/api.js";
import { app } from "../../../../scripts/app.js";

// Helper for getting/setting values in objects using path strings
const PathHelper = {
    get(obj, path) {
        if (typeof path !== "string") {
            return path;
        }

        if (path[0] === '"' && path[path.length - 1] === '"') {
            return JSON.parse(path);
        }
        console.log('[Debug] PathHelper.get called with:', { obj, path });

        const parts = path.split(".").filter(Boolean);
        for (const p of parts) {
            if (obj == null) {
                console.warn(`[Debug] Property "${p}" not found in chain for path "${parts.join('.')}"`);
                return undefined;
            }
            // Check if current object is an array
            if (Array.isArray(obj)) {
                // Try to find an element with property name === p
                obj = obj.find(el => el.name === p);
                if (obj === undefined) {
                    console.warn(`[Debug] Could not find widget with name "${p}" in array.`);
                    return undefined;
                }
            } else {
                const k = isNaN(+p) ? p : +p;
                obj = obj[k];
            }
        }

        return obj;
    },
    set(obj, path, value) {
        if (Object(obj) !== obj) return obj;
        if (!Array.isArray(path)) path = path.toString().match(/[^.[\]]+/g) || [];
        path.slice(0, -1).reduce(
            (a, c, i) =>
                Object(a[c]) === a[c]
                    ? a[c]
                    : (a[c] =
                          Math.abs(path[i + 1]) >> 0 === +path[i + 1]
                              ? []
                              : {}),
            obj
        )[path[path.length - 1]] = value;
        return obj;
    },
};

// Function to evaluate conditions for if callbacks
function evaluateCondition(condition, state) {
    const left = PathHelper.get(state, condition.left);
    const right = PathHelper.get(state, condition.right);

    if (condition.op === "eq") {
        return left === right;
    } else {
        return left !== right;
    }
}

// Callback implementations
const callbacks = {
    async if(cb, state) {
        let success = true;
        for (const condition of cb.condition) {
            const r = evaluateCondition(condition, state);
            if (!r) {
                success = false;
                break;
            }
        }

        for (const m of cb[success + ""] ?? []) {
            await invokeCallback(m, state);
        }
    },

    async fetch(cb, state) {
        const url = cb.url.replace(/\{([^\}]+)\}/g, (m, v) => {
            return PathHelper.get(state, v);
        });
        const res = await (await api.fetchApi(url)).json();
        state["$result"] = res;
        for (const m of cb.then) {
            await invokeCallback(m, state);
        }
    },

    async set(cb, state) {
        const value = PathHelper.get(state, cb.value);
        PathHelper.set(state, cb.target, value);
    },

    async "validate-combo"(cb, state) {
        const w = state["$this"];
        const valid = w.options.values.includes(w.value);
        if (!valid) {
            w.value = w.options.values[0];
        }
    },
};

async function invokeCallback(callback, state) {
    if (callback.type in callbacks) {
        await callbacks[callback.type](callback, state);
    } else {
        console.warn(
            "%c[🪄 Plush]",
            "color: purple",
            `[binding ${state.$node.comfyClass}.${state.$this.name}]`,
            "unsupported binding callback type:",
            callback.type
        );
    }
}

// Register the binding extension
app.registerExtension({
    name: "plush.Binding",
    beforeRegisterNodeDef(node, nodeData) {
        const hasBinding = (v) => {
            if (!v) return false;
            return Object.values(v).find((c) => c[1]?.["plush.binding"]);
        };
        
        const inputs = { ...nodeData.input?.required, ...nodeData.input?.optional };
        if (hasBinding(inputs)) {
            const onAdded = node.prototype.onAdded;
            node.prototype.onAdded = function () {
                const r = onAdded?.apply(this, arguments);

                for (const widget of this.widgets || []) {
                    const bindings = inputs[widget.name][1]?.["plush.binding"];
                    if (!bindings) continue;

                    for (const binding of bindings) {
                        const source = this.widgets.find((w) => w.name === binding.source);
                        if (!source) {
                            console.warn(
                                "%c[🪄 Plush]",
                                "color: purple",
                                `[binding ${node.comfyClass}.${widget.name}]`,
                                "unable to find source binding widget:",
                                binding.source,
                                binding
                            );
                            continue;6
                        }

                        let lastValue;
                        async function valueChanged() {
                            console.log('[Debug] valueChanged triggered for widget: ', widget.name,  'with new value: ', source.value)
                            const state = {
                                $this: widget,
                                $source: source,
                                $node: node,
                            };

                            console.log('[Debug] state.$node:', state.$node);
                            console.log('[Debug] state.$node.widgets:', state.$node.widgets);
                            console.log('[Debug] Full state.$node:', state.$node);

                            console.log('[Debug] state.$this:', state.$this);
                            console.log('[Debug] state.$this.parent:', state.$this.parent);
                            console.log('[Debug] state.$this.parent.widgets:', state.$this.parent ? state.$this.parent.widgets : 'No parent');

                            for (const callback of binding.callback) {
                                await invokeCallback(callback, state);
                            }

                            app.graph.setDirtyCanvas(true, false);
                        }

                        const cb = source.callback;
                        source.callback = function () {
                            const v = cb?.apply(this, arguments) ?? source.value;
                            if (v !== lastValue) {
                                lastValue = v;
                                valueChanged();
                            }
                            return v;
                        };

                        lastValue = source.value;
                        valueChanged();
                    }
                }

                return r;
            };
        }
    },
});