// Reusable confirmation dialog component for user action confirmations
import { AlertTriangle, X } from 'lucide-react';

// Props interface for the ConfirmationDialog component
interface ConfirmationDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmText: string;
  cancelText: string;
  onConfirm: () => void;
  onCancel: () => void;
  type?: 'danger' | 'warning' | 'info';
}

// Main ConfirmationDialog component with customizable styling based on dialog type
export default function ConfirmationDialog({
  isOpen,
  title,
  message,
  confirmText,
  cancelText,
  onConfirm,
  onCancel,
  type = 'warning'
}: ConfirmationDialogProps) {
  // Early return if dialog is not open
  if (!isOpen) return null;

  // Function to get styling based on dialog type (danger, warning, info)
  const getTypeStyles = () => {
    switch (type) {
      case 'danger':
        return {
          icon: 'text-red-600',
          confirmButton: 'bg-red-600 hover:bg-red-700 text-white',
          border: 'border-red-200'
        };
      case 'warning':
        return {
          icon: 'text-yellow-600',
          confirmButton: 'bg-yellow-600 hover:bg-yellow-700 text-white',
          border: 'border-yellow-200'
        };
      default:
        return {
          icon: 'text-blue-600',
          confirmButton: 'bg-blue-600 hover:bg-blue-700 text-white',
          border: 'border-blue-200'
        };
    }
  };

  const styles = getTypeStyles();

  return (
    // Modal overlay covering the entire screen
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      {/* Dialog container with dynamic styling */}
      <div className={`bg-white rounded-lg shadow-xl max-w-md w-full mx-4 border-2 ${styles.border}`}>
        <div className="p-6">
          {/* Header section with icon, title, and close button */}
          <div className="flex items-center mb-4">
            <AlertTriangle className={`w-6 h-6 mr-3 ${styles.icon}`} />
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
            <button
              onClick={onCancel}
              className="ml-auto text-gray-400 hover:text-gray-600"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          {/* Message content */}
          <p className="text-gray-700 mb-6">{message}</p>
          
          {/* Action buttons section */}
          <div className="flex space-x-3 justify-end">
            {/* Cancel button */}
            <button
              onClick={onCancel}
              className="px-4 py-2 text-gray-700 bg-gray-200 hover:bg-gray-300 rounded-lg font-medium transition-colors"
            >
              {cancelText}
            </button>
            {/* Confirm button with dynamic styling */}
            <button
              onClick={onConfirm}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${styles.confirmButton}`}
            >
              {confirmText}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}