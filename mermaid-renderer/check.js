// The claim this image makes, asserted where a failure stops the build: a
// browser that starts, and a real diagram that comes back as real SVG.
import { createRenderer } from "./render.js";

const renderer = await createRenderer();
try {
  const svg = (
    await renderer.render(
      "graph TD\n  A[ingest] --> B[embed]\n  B --> C[(qdrant)]\n",
      "svg",
    )
  ).toString("utf8");
  if (!svg.includes("<svg") || !svg.includes("qdrant")) {
    throw new Error(`rendered SVG lacks the diagram: ${svg.slice(0, 200)}`);
  }
  const png = await renderer.render("graph LR\n  A[render] --> B[png]\n", "png");
  // A PNG is worth rendering here as well as an SVG: the SVG carries text as
  // characters, so a missing font cannot show up in it, while the PNG is the
  // path where a font failure becomes empty boxes.
  if (png.length < 1000 || png.subarray(1, 4).toString("ascii") !== "PNG") {
    throw new Error(`PNG render produced ${png.length} bytes of something else`);
  }
  console.log(
    `render check ok: ${svg.length} characters of SVG, ${png.length} bytes of PNG`,
  );
} finally {
  await renderer.close();
}
