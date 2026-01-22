/**
 * SlidePreview Component
 *
 * Renders a single presentation slide with theme styling.
 * Supports all slide types: title, content, diagram, summary, two_column.
 */

import React from 'react';
import { PresentationSlideContent, ThemeConfig, PresentationStyle } from '../../types';
import { getThemeForPresentation } from './themeDefaults';
import MermaidDiagram from '../common/MermaidDiagram';

interface SlidePreviewProps {
  slide: PresentationSlideContent;
  theme?: ThemeConfig | null;
  style?: PresentationStyle;
  showNotes?: boolean;
}

const SlidePreview: React.FC<SlidePreviewProps> = ({
  slide,
  theme,
  style = 'professional',
  showNotes = false,
}) => {
  const activeTheme = getThemeForPresentation(style, theme);
  const { colors, fonts } = activeTheme;

  // Base slide container with 16:9 aspect ratio
  const slideContainerStyle: React.CSSProperties = {
    backgroundColor: colors.bg_color,
    fontFamily: fonts.body_font,
    aspectRatio: '16 / 9',
    position: 'relative',
    overflow: 'hidden',
  };

  const renderTitleSlide = () => (
    <div
      className="flex flex-col items-center justify-center h-full p-8 text-center"
      style={slideContainerStyle}
    >
      <h1
        className="text-4xl md:text-5xl font-bold mb-4"
        style={{ color: colors.title_color, fontFamily: fonts.title_font }}
      >
        {slide.title}
      </h1>
      {slide.subtitle && (
        <p
          className="text-xl md:text-2xl mb-6"
          style={{ color: colors.text_color }}
        >
          {slide.subtitle}
        </p>
      )}
      {!slide.subtitle && slide.content && slide.content[0] && (
        <p
          className="text-xl md:text-2xl mb-6"
          style={{ color: colors.text_color }}
        >
          {slide.content[0]}
        </p>
      )}
      <div
        className="w-32 h-1 rounded"
        style={{ backgroundColor: colors.accent_color }}
      />
    </div>
  );

  const renderContentSlide = () => (
    <div className="flex flex-col h-full p-6 md:p-8" style={slideContainerStyle}>
      {/* Header */}
      <div className="mb-4">
        <h2
          className="text-2xl md:text-3xl font-bold mb-2"
          style={{ color: colors.title_color, fontFamily: fonts.title_font }}
        >
          {slide.title}
        </h2>
        <div
          className="w-16 h-1 rounded"
          style={{ backgroundColor: colors.accent_color }}
        />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        <ul className="space-y-3">
          {slide.content.map((point, i) => (
            <li key={i} className="flex items-start">
              <span
                className="mr-3 text-xl font-bold flex-shrink-0"
                style={{ color: colors.accent_color }}
              >
                &bull;
              </span>
              <span
                className="text-base md:text-lg"
                style={{ color: colors.text_color }}
              >
                {point}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );

  const renderDiagramSlide = () => (
    <div className="flex flex-col h-full p-6 md:p-8" style={slideContainerStyle}>
      {/* Header */}
      <div className="mb-4">
        <h2
          className="text-2xl md:text-3xl font-bold mb-2"
          style={{ color: colors.title_color, fontFamily: fonts.title_font }}
        >
          {slide.title}
        </h2>
      </div>

      {/* Diagram */}
      <div className="flex-1 flex items-center justify-center overflow-hidden">
        {slide.diagram_code ? (
          <div className="w-full max-h-full">
            <MermaidDiagram
              code={slide.diagram_code}
              title={slide.title}
              className="border-0 shadow-none"
            />
          </div>
        ) : slide.diagram_description ? (
          <div className="text-center">
            <div className="w-64 h-40 mx-auto border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center mb-4">
              <svg
                className="w-16 h-16 text-gray-300"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"
                />
              </svg>
            </div>
            <p className="text-gray-400 italic text-sm">
              {slide.diagram_description}
            </p>
          </div>
        ) : (
          <p className="text-gray-400 italic">[Diagram not available]</p>
        )}
      </div>
    </div>
  );

  const renderSummarySlide = () => (
    <div className="flex flex-col h-full p-6 md:p-8" style={slideContainerStyle}>
      {/* Header */}
      <div className="mb-4">
        <h2
          className="text-2xl md:text-3xl font-bold mb-2"
          style={{ color: colors.title_color, fontFamily: fonts.title_font }}
        >
          {slide.title}
        </h2>
        <div
          className="w-full h-1 rounded"
          style={{ backgroundColor: colors.accent_color }}
        />
      </div>

      {/* Summary points with checkmarks */}
      <div className="flex-1 overflow-auto">
        <ul className="space-y-4">
          {slide.content.map((point, i) => (
            <li key={i} className="flex items-start">
              <span
                className="mr-3 text-xl flex-shrink-0"
                style={{ color: colors.accent_color }}
              >
                &#10003;
              </span>
              <span
                className="text-base md:text-lg"
                style={{ color: colors.text_color }}
              >
                {point}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );

  const renderTwoColumnSlide = () => {
    const midpoint = Math.ceil(slide.content.length / 2);
    const leftColumn = slide.content.slice(0, midpoint);
    const rightColumn = slide.content.slice(midpoint);

    return (
      <div className="flex flex-col h-full p-6 md:p-8" style={slideContainerStyle}>
        {/* Header */}
        <div className="mb-4">
          <h2
            className="text-2xl md:text-3xl font-bold mb-2"
            style={{ color: colors.title_color, fontFamily: fonts.title_font }}
          >
            {slide.title}
          </h2>
          <div
            className="w-16 h-1 rounded"
            style={{ backgroundColor: colors.accent_color }}
          />
        </div>

        {/* Two columns */}
        <div className="flex-1 grid grid-cols-2 gap-6 overflow-auto">
          <ul className="space-y-2">
            {leftColumn.map((point, i) => (
              <li key={i} className="flex items-start">
                <span
                  className="mr-2 text-lg font-bold flex-shrink-0"
                  style={{ color: colors.accent_color }}
                >
                  &bull;
                </span>
                <span
                  className="text-sm md:text-base"
                  style={{ color: colors.text_color }}
                >
                  {point}
                </span>
              </li>
            ))}
          </ul>
          <ul className="space-y-2">
            {rightColumn.map((point, i) => (
              <li key={i} className="flex items-start">
                <span
                  className="mr-2 text-lg font-bold flex-shrink-0"
                  style={{ color: colors.accent_color }}
                >
                  &bull;
                </span>
                <span
                  className="text-sm md:text-base"
                  style={{ color: colors.text_color }}
                >
                  {point}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  };

  const renderSlide = () => {
    switch (slide.slide_type) {
      case 'title':
        return renderTitleSlide();
      case 'content':
        return renderContentSlide();
      case 'diagram':
        return renderDiagramSlide();
      case 'summary':
        return renderSummarySlide();
      case 'two_column':
        return renderTwoColumnSlide();
      default:
        return renderContentSlide();
    }
  };

  return (
    <div className="w-full">
      {/* Slide */}
      <div className="rounded-lg shadow-lg overflow-hidden border border-gray-200">
        {renderSlide()}
      </div>

      {/* Speaker Notes */}
      {showNotes && slide.notes && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Speaker Notes</h4>
          <p className="text-sm text-gray-600">{slide.notes}</p>
        </div>
      )}
    </div>
  );
};

export default SlidePreview;
