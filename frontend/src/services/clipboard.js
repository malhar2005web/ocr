/**
 * Copies text to the clipboard with full fallback support for:
 * 1. Modern secure contexts (navigator.clipboard)
 * 2. Non-secure HTTP contexts (e.g. http://173.249.59.181:3050 where navigator.clipboard is undefined)
 * 3. Mobile browsers and WebViews
 */
export async function copyTextToClipboard(text) {
  if (!text) return false;

  // 1. Try modern Async Clipboard API if available and context is secure
  if (navigator?.clipboard && typeof navigator.clipboard.writeText === 'function') {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      console.warn('navigator.clipboard.writeText failed, attempting execCommand fallback:', err);
    }
  }

  // 2. Fallback using invisible textarea and document.execCommand('copy')
  try {
    const textArea = document.createElement('textarea');
    textArea.value = text;

    // Ensure the textarea is off-screen and does not cause layout jump
    textArea.style.position = 'fixed';
    textArea.style.top = '0';
    textArea.style.left = '-9999px';
    textArea.style.width = '2em';
    textArea.style.height = '2em';
    textArea.style.padding = '0';
    textArea.style.border = 'none';
    textArea.style.outline = 'none';
    textArea.style.boxShadow = 'none';
    textArea.style.background = 'transparent';
    textArea.setAttribute('readonly', '');

    document.body.appendChild(textArea);
    textArea.select();
    textArea.setSelectionRange(0, text.length);

    const successful = document.execCommand('copy');
    document.body.removeChild(textArea);
    return successful;
  } catch (err) {
    console.error('All clipboard copy methods failed:', err);
    return false;
  }
}
