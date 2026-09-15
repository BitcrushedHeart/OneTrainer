import { describe, expect, it } from "vitest";

import { DEFAULT_SCORE_WEIGHTS, sortByCombined, weightedZ } from "../scoreCombine";

const images = {
  best: { text: null, adherence: 0.9, age: 0.9, realism: 0.1, quality: 80 },
  mid: { text: null, adherence: 0.6, age: 0.6, realism: 0.4, quality: 50 },
  worst: { text: null, adherence: 0.3, age: 0.3, realism: 0.9, quality: 20 },
};

describe("weightedZ", () => {
  it("is sign-corrected: lower P(AI) scores higher", () => {
    const z = weightedZ(images, DEFAULT_SCORE_WEIGHTS);
    expect(z.best).toBeGreaterThan(z.mid);
    expect(z.mid).toBeGreaterThan(z.worst);
    const realismOnly = weightedZ(images, { text: 0, adherence: 0, age: 0, realism: 1, quality: 0 });
    expect(realismOnly.best).toBeGreaterThan(realismOnly.worst);
  });

  it("lets the default text weight outweigh every other axis together", () => {
    const pair = {
      rightText: { text: 1, adherence: 0.5, age: 0.5, realism: 0.6, quality: 40 },
      wrongText: { text: 0.5, adherence: 0.8, age: 0.8, realism: 0.2, quality: 70 },
    };
    const z = weightedZ(pair, DEFAULT_SCORE_WEIGHTS);
    expect(z.rightText).toBeGreaterThan(z.wrongText);
    const { text, ...rest } = DEFAULT_SCORE_WEIGHTS;
    expect(text).toBeGreaterThan(Object.values(rest).reduce((a, b) => a + b, 0));
  });

  it("lets a prompt focus on age and adherence over realism", () => {
    const pair = {
      a: { text: null, adherence: 0.9, age: 0.9, realism: 0.9, quality: 10 },
      b: { text: null, adherence: 0.3, age: 0.3, realism: 0.1, quality: 90 },
    };
    const focus = weightedZ(pair, { text: 0, adherence: 1, age: 1, realism: 0, quality: 0 });
    expect(focus.a).toBeGreaterThan(focus.b);
    const realism = weightedZ(pair, { text: 0, adherence: 0, age: 0, realism: 1, quality: 0 });
    expect(realism.b).toBeGreaterThan(realism.a);
  });

  it("ignores missing axes and a constant axis without NaN", () => {
    const z = weightedZ(
      {
        a: { text: null, adherence: 0.5, age: null, realism: 0.2, quality: 50 },
        b: { text: null, adherence: 0.5, age: null, realism: 0.2, quality: 50 },
        c: { text: undefined, adherence: undefined, age: undefined, realism: undefined, quality: undefined },
      },
      DEFAULT_SCORE_WEIGHTS,
    );
    expect(z.a).toBe(0);
    expect(z.b).toBe(0);
    expect(z.c).toBe(0);
  });
});

describe("sortByCombined", () => {
  it("orders best-first and keeps unscored paths last in their own order", () => {
    const order = sortByCombined(["worst", "x", "best", "y", "mid"], images, DEFAULT_SCORE_WEIGHTS);
    expect(order).toEqual(["best", "mid", "worst", "x", "y"]);
  });
});
