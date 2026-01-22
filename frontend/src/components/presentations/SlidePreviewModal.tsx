/**
 * SlidePreviewModal Component
 *
 * Full-screen modal for previewing presentation slides with navigation.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { PresentationJob } from '../../types';
import SlidePreview from './SlidePreview';
import { getThemeForPresentation } from './themeDefaults';

interface SlidePreviewModalProps {
  job: PresentationJob;
  isOpen: boolean;
  onClose: () => void;
}

const SlidePreviewModal: React.FC<SlidePreviewModalProps> = ({
  job,
  isOpen,
  onClose,
}) => {
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [showNotes, setShowNotes] = useState(false);

  const slides = job.generated_outline?.slides || [];
  const totalSlides = slides.length;
  const currentSlide = slides[currentSlideIndex];

  // Navigation functions
  const goToSlide = useCallback((index: number) => {
    if (index >= 0 && index < totalSlides) {
      setCurrentSlideIndex(index);
    }
  }, [totalSlides]);

  const nextSlide = useCallback(() => {
    goToSlide(currentSlideIndex + 1);
  }, [currentSlideIndex, goToSlide]);

  const prevSlide = useCallback(() => {
    goToSlide(currentSlideIndex - 1);
  }, [currentSlideIndex, goToSlide]);

  const firstSlide = useCallback(() => {
    goToSlide(0);
  }, [goToSlide]);

  const lastSlide = useCallback(() => {
    goToSlide(totalSlides - 1);
  }, [totalSlides, goToSlide]);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowRight':
        case 'ArrowDown':
        case ' ':
          e.preventDefault();
          nextSlide();
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          e.preventDefault();
          prevSlide();
          break;
        case 'Home':
          e.preventDefault();
          firstSlide();
          break;
        case 'End':
          e.preventDefault();
          lastSlide();
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, nextSlide, prevSlide, firstSlide, lastSlide, onClose]);

  // Reset to first slide when modal opens
  useEffect(() => {
    if (isOpen) {
      setCurrentSlideIndex(0);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  if (!slides.length) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50">
        <div className="text-center text-white">
          <svg
            className="mx-auto h-16 w-16 text-gray-400 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h3 className="text-xl font-medium mb-2">No slides available</h3>
          <p className="text-gray-400 mb-4">
            The presentation outline hasn't been generated yet.
          </p>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white text-gray-900 rounded-md hover:bg-gray-100"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-95 flex flex-col z-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-900 text-white">
        <div className="flex items-center">
          <button
            onClick={onClose}
            className="mr-4 p-1 hover:bg-gray-700 rounded"
            title="Close (Esc)"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <div>
            <h2 className="font-semibold truncate max-w-md">{job.title}</h2>
            <p className="text-xs text-gray-400">
              {job.generated_outline?.title || job.topic}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Notes toggle */}
          <button
            onClick={() => setShowNotes(!showNotes)}
            className={`flex items-center px-3 py-1.5 rounded text-sm ${
              showNotes ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
            title="Toggle speaker notes"
          >
            <svg className="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            Notes
          </button>

          {/* Close button */}
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-gray-700 rounded"
            title="Close"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex items-center justify-center p-4 md:p-8 overflow-hidden">
        {/* Previous button */}
        <button
          onClick={prevSlide}
          disabled={currentSlideIndex === 0}
          className={`flex-shrink-0 p-2 md:p-4 rounded-full mr-2 md:mr-4 transition-colors ${
            currentSlideIndex === 0
              ? 'text-gray-600 cursor-not-allowed'
              : 'text-white hover:bg-gray-700'
          }`}
          title="Previous slide (Left arrow)"
        >
          <svg className="w-8 h-8 md:w-10 md:h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        {/* Slide preview container */}
        <div className="flex-1 max-w-5xl max-h-full overflow-auto">
          {currentSlide && (
            <SlidePreview
              slide={currentSlide}
              style={job.style}
              showNotes={showNotes}
            />
          )}
        </div>

        {/* Next button */}
        <button
          onClick={nextSlide}
          disabled={currentSlideIndex === totalSlides - 1}
          className={`flex-shrink-0 p-2 md:p-4 rounded-full ml-2 md:ml-4 transition-colors ${
            currentSlideIndex === totalSlides - 1
              ? 'text-gray-600 cursor-not-allowed'
              : 'text-white hover:bg-gray-700'
          }`}
          title="Next slide (Right arrow)"
        >
          <svg className="w-8 h-8 md:w-10 md:h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>

      {/* Footer with slide indicator and thumbnails */}
      <div className="bg-gray-900 text-white px-4 py-3">
        {/* Slide counter */}
        <div className="text-center mb-3">
          <span className="text-lg font-medium">
            {currentSlideIndex + 1} / {totalSlides}
          </span>
          {currentSlide && (
            <span className="text-gray-400 text-sm ml-2">
              - {currentSlide.title}
            </span>
          )}
        </div>

        {/* Slide thumbnails */}
        <div className="flex justify-center space-x-2 overflow-x-auto pb-2">
          {slides.map((slide, index) => (
            <button
              key={index}
              onClick={() => goToSlide(index)}
              className={`flex-shrink-0 w-16 h-10 rounded border-2 transition-all ${
                index === currentSlideIndex
                  ? 'border-blue-500 ring-2 ring-blue-500 ring-opacity-50'
                  : 'border-gray-600 hover:border-gray-400'
              }`}
              title={`${index + 1}. ${slide.title}`}
            >
              <div
                className="w-full h-full rounded flex items-center justify-center text-xs"
                style={{
                  backgroundColor: getThemeForPresentation(job.style).colors.bg_color,
                  color: getThemeForPresentation(job.style).colors.text_color,
                }}
              >
                {index + 1}
              </div>
            </button>
          ))}
        </div>

        {/* Keyboard shortcuts hint */}
        <div className="text-center text-xs text-gray-500 mt-2">
          Use arrow keys to navigate &bull; Press Esc to close
        </div>
      </div>
    </div>
  );
};

export default SlidePreviewModal;
