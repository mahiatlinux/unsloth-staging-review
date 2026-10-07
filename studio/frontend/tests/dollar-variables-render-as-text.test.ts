// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createMathPlugin } from "@streamdown/math";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";
import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";
import { normalizeEscapedInlineMath } from "../src/lib/escaped-inline-math.ts";
import { preprocessLaTeX } from "../src/lib/latex.ts";

const math = createMathPlugin({ singleDollarTextMath: true });

const HTML_ENTITY_TEXT: Record<string, string> = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#x27;": "'",
};

function render(source: string, isStreaming = false): string {
  return renderToStaticMarkup(
    React.createElement(
      Streamdown,
      {
        mode: "streaming",
        isAnimating: isStreaming,
        plugins: { math },
      },
      stabilizeStreamingMarkdown(
        preprocessLaTeX(normalizeEscapedInlineMath(source)),
        isStreaming,
      ),
    ),
  );
}

function renderedMath(html: string): string[] {
  return [
    ...html.matchAll(
      /<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g,
    ),
  ].map((match) =>
    match[1].replace(
      /&(?:amp|lt|gt|quot|#x27);/g,
      (entity) => HTML_ENTITY_TEXT[entity] ?? entity,
    ),
  );
}

const SHELL_VARIABLE_REPLIES: Array<[string, string[]]> = [
  [
    "Add the bin folder to $PATH, then check $HOME/.bashrc.",
    ["$PATH, then check $HOME/.bashrc."],
  ],
  [
    "Export $CUDA_HOME and $LD_LIBRARY_PATH before building.",
    ["$CUDA_HOME and $LD_LIBRARY_PATH"],
  ],
  ["Run echo $PATH\nthen echo $HOME", ["$PATH", "$HOME"]],
  ["echo $PATH\n| sed 's/bin/sbin/'\necho $HOME", ["$PATH", "$HOME"]],
  ["Run echo $PATH $HOME", ["$PATH $HOME"]],
  ["echo $HOME$USER", ["$HOME$USER"]],
  ["cp $SRC$SUFFIX $DST", ["$SRC$SUFFIX $DST"]],
  ["cp $src$dst $out", ["$src$dst $out"]],
  ["Use $PATH **or** $HOME", ["$PATH", "$HOME"]],
  ["Use $PATH, then **$HOME**", ["$PATH, then", "$HOME"]],
  ["Use $PATH, then ~~$HOME~~", ["$PATH, then", "$HOME"]],
  ["Use $PATH, then [$HOME](https://example.com)", ["$PATH, then", "$HOME"]],
  [
    "Use $PATH or [the docs](https://example.com), then $HOME",
    ["$PATH", "$HOME"],
  ],
  ["cp $HOME/src $PATH/bin", ["$HOME/src", "$PATH/bin"]],
  ["cp $src $dst", ["$src", "$dst"]],
  ["Status is $?; home is $HOME", ["$?", "$HOME"]],
  ["Count is $#; home is $HOME", ["$#", "$HOME"]],
  ["Args are $@; home is $HOME", ["$@", "$HOME"]],
  ["PID is $!; home is $HOME", ["$!", "$HOME"]],
  ["All args are $*; home is $HOME", ["$*", "$HOME"]],
  ["Flags are $-; home is $HOME", ["$-", "$HOME"]],
  ["home is $HOME; status is $?", ["$HOME", "$?"]],
  ["args are $PATH then $@", ["$PATH", "$@"]],
  ["echo $$; kill $$", ["$$", "$$"]],
  [
    "In Bash, $$ is the PID and $$ is inherited by subshells.",
    ["$$ is the PID and $$ is inherited by subshells."],
  ],
  ["使用 $PATH，然后 $HOME。", ["$PATH", "$HOME"]],
  ["Use $PATH — or $HOME", ["$PATH", "$HOME"]],
  ["Check $HOME, $PATH, and $USER first.", ["$HOME, $PATH, and $USER"]],
  ['Quote them: "$HOME" and "$PATH".', ["$HOME", "$PATH"]],
  ["Files live under $HOME/$USER/data.", ["$HOME/$USER/data"]],
  ["Use ${HOME} and ${PATH} in scripts.", ["${HOME} and ${PATH}"]],
  [
    "Use ${HOME:-/tmp}, then ${PATH:-/usr/bin}",
    ["${HOME:-/tmp}", "${PATH:-/usr/bin}"],
  ],
  [
    "Use ${HOME%/*}, ${PATH#*/}, and ${value/pat/repl}; then ${USER%/*}",
    ["${HOME%/*}", "${PATH#*/}", "${value/pat/repl}", "${USER%/*}"],
  ],
  ["echo ${FOO:-$HOME} then ${BAR:+$PATH}", ["${FOO:-$HOME}", "${BAR:+$PATH}"]],
  ["echo ${HOME}$PATH", ["${HOME}$PATH"]],
  [
    "Use ${!prefix*}, ${value:1:3}, and ${array[0]}; then $HOME",
    ["${!prefix*}", "${value:1:3}", "${array[0]}", "$HOME"],
  ],
  [
    "echo ${!name} ${#array[@]} ${!array[*]}",
    ["${!name}", "${#array[@]}", "${!array[*]}"],
  ],
  [
    'echo "${GREETING:-hello world}" "${NAME:-John Doe}"',
    ["${GREETING:-hello world}", "${NAME:-John Doe}"],
  ],
  ["Set it to $PATH:$HOME/bin now.", ["$PATH:$HOME/bin"]],
  ["#define HOME $HOME\nthen $PATH", ["$HOME", "$PATH"]],
  ["Use $PATH\n***\nthen $HOME and $USER", ["$PATH", "$HOME and $USER"]],
  [
    "Try $PATH || $HOME, $HOME > $LOG or $PATH=$HOME/bin.",
    ["$PATH || $HOME", "$PATH=$HOME/bin"],
  ],
  ["PHP reads $_GET and $_POST.", ["$_GET and $"]],
  [
    "Add it to $PATH, then run `echo $HOME`.",
    ["$PATH, then run", "echo $HOME"],
  ],
];

test("shell variables in a reply render as text, not maths", () => {
  for (const [source, texts] of SHELL_VARIABLE_REPLIES) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), [], source);
    for (const text of texts) {
      assert.ok(html.includes(text), `${source} lost ${text}`);
    }
  }
  assert.ok(
    render("Use $PATH or [the docs](https://example.com), then $HOME").includes(
      'data-streamdown="link"',
    ),
  );
});

test("shell variables never flash as maths while a reply streams", () => {
  for (const [source] of SHELL_VARIABLE_REPLIES) {
    for (let end = 1; end <= source.length; end += 1) {
      const prefix = source.slice(0, end);
      assert.deepEqual(renderedMath(render(prefix, true)), [], prefix);
    }
  }
});

const MATH_REPLIES: Array<[string, string[]]> = [
  ["$x$", ["x"]],
  ["In triangle $ABC$, the side $AB = 5$.", ["ABC", "AB = 5"]],
  ["The slope is $dy/dx$ and the line is $ax + b$.", ["dy/dx", "ax + b"]],
  ["Use $\\alpha + 1$ and $\\alpha$.", ["\\alpha + 1", "\\alpha"]],
  ["Lines $AB$ and $CD$ meet at $P$.", ["AB", "CD", "P"]],
  ["Segments $AB and CD$ are equal.", ["AB and CD"]],
  ["The $n$th element and the $k$th one.", ["n", "k"]],
  [
    "We have $sin x$ and $n log n$ and $sin theta$.",
    ["sin x", "n log n", "sin theta"],
  ],
  ["Points $x_1, x_2$ and $v_s$ and $a_{ij}$.", ["x_1, x_2", "v_s", "a_{ij}"]],
  ["So $E = mc^2$ and $f(x)$ and $O(n)$.", ["E = mc^2", "f(x)", "O(n)"]],
  ["Then $P(A and B)$ holds.", ["P(A and B)"]],
  ["Pick ${n \\choose k}$ ways.", ["{n \\choose k}"]],
  ["Write $ab/cd$ or $AB/CD$.", ["ab/cd", "AB/CD"]],
  ["Then $\\text{if } x > 0$ and $|x|$.", ["\\text{if } x > 0", "|x|"]],
  ["An angle of $30^\\circ$ here.", ["30^\\circ"]],
  ["**$90 - x$** is the rest.", ["90 - x"]],
  ["Let $x \\in A$ and $AB $ hold.", ["x \\in A", "AB "]],
  ["The value \\(\\beta\\) and $\\gamma$.", ["\\beta", "\\gamma"]],
  ["Sum $a +\nb$ over lines.", ["a +\nb"]],
  ["Segment $AB'$ and the derivative $uv'$.", ["AB'", "uv'"]],
  [
    "About $\\sim$2x faster: the $k$th token and the $n$th layer.",
    ["\\sim", "k", "n"],
  ],
  [
    "| Item | Price ($) |\n|---|---|\n| the $n$th row and the $m$th column | 5 |",
    ["n", "m"],
  ],
  ["# Cost in $\nThe $n$th row and the $m$th column.", ["n", "m"]],
  ["- Ends with $\n- The $n$th row and the $m$th column", ["n", "m"]],
  ["We have $sin theta $ and $AB / CD $ here.", ["sin theta ", "AB / CD "]],
  ["The form $sin theta $is useful.", ["sin theta "]],
  ["The ratio $AB / CD $is useful.", ["AB / CD "]],
  [
    "The relation $velocity = distance / time $is useful.",
    ["velocity = distance / time "],
  ],
  ["The result is $velocity(t) = distance $", ["velocity(t) = distance "]],
  ["The probability is $softmax(x) $", ["softmax(x) "]],
  ["Let $theta $be positive and $radius $stay finite.", ["theta ", "radius "]],
  ["Segments $AB, CD $are congruent.", ["AB, CD "]],
  ["The triangles are $ABC, DEF $", ["ABC, DEF "]],
  ["Parameters $alpha, beta $", ["alpha, beta "]],
  ["Rates $distance/time, speed/time $", ["distance/time, speed/time "]],
  ["In triangle $ABC $is acute.", ["ABC "]],
  ["Use $softmax(x), sigmoid(x) $for the logits.", ["softmax(x), sigmoid(x) "]],
  [
    "The map $softmax(sigmoid(x)) $is continuous.",
    ["softmax(sigmoid(x)) "],
  ],
  ["Source $theta $is the angle.", ["theta "]],
  ["Export $radius $as CSV.", ["radius "]],
  ["The result is $det A $for this matrix.", ["det A "]],
  ["The product is $theta phi $in the basis.", ["theta phi "]],
  ["The result is $det A $", ["det A "]],
  ["The product is $theta phi $", ["theta phi "]],
  ["The product is $alpha beta gamma $", ["alpha beta gamma "]],
  ["Product ${a*b}$.", ["{a*b}"]],
  ["Braced ${x^2}$ and ${a+b}$.", ["{x^2}", "{a+b}"]],
  ["Convolution is $*$ and $*x$.", ["*", "*x"]],
  ["Use $HOME/$USER and $\\alpha$ here.", ["\\alpha"]],
];

test("real maths still renders", () => {
  for (const [source, expected] of MATH_REPLIES) {
    assert.deepEqual(renderedMath(render(source)), expected, source);
  }
});

test("braced maths is not rewritten as shell parameter markup", () => {
  for (const [source, expected] of [
    ["${a*b}$", "a*b"],
    ["\\(${a*b}\\)", "a*b"],
    ["\\(${x^2}\\)", "x^2"],
    ["The `export` example leaves ${x}$ unchanged.", "{x}"],
    ["The export example leaves ${x}$ unchanged.", "{x}"],
  ]) {
    const preprocessed = preprocessLaTeX(source);
    assert.ok(preprocessed.includes(expected), source);
    assert.ok(!preprocessed.includes("&#42;"), source);
    assert.ok(!preprocessed.includes("&#36;"), source);
  }
  assert.equal(preprocessLaTeX("Display $$x$$."), "Display $$x$$.");
  assert.equal(
    preprocessLaTeX("After a short wait, $$\nx^2\n$$"),
    "After a short wait, $$\nx^2\n$$",
  );
});

test("escaped shell markup stays escaped", () => {
  for (const source of ["\\${HOME}", "\\$*"]) {
    assert.equal(preprocessLaTeX(source), source);
  }
});

test("maths and currency next to shell variables keep rendering", () => {
  const withMath = render("Set $x$ from $PATH, then $HOME.");
  assert.deepEqual(renderedMath(withMath), ["x"]);
  assert.ok(withMath.includes("$PATH, then $HOME."));

  const withCurrency = render("It costs $5 to set $PATH, then $HOME.");
  assert.deepEqual(renderedMath(withCurrency), []);
  assert.ok(withCurrency.includes("$5 to set $PATH, then $HOME."));
});

test("shell variables keep literal dollars in raw HTML and URLs", () => {
  for (const source of [
    "<pre>$HOME and $USER</pre>",
    "https://example.com/$HOME/$USER",
    "<https://example.com/$HOME/$USER>",
  ]) {
    const html = render(source);
    assert.deepEqual(renderedMath(html), [], source);
    assert.ok(html.includes("$HOME"), `${source} lost its first dollar`);
    assert.ok(!html.includes("\\$HOME"), `${source} showed an escape slash`);
  }
});
