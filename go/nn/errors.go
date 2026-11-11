package nn

import "errors"

// ErrGeneratedErrors is returned when the errors operation cannot be completed.
var ErrGeneratedErrors = errors.New("pygo: errors operation failed")

// GeneratedErrorsError wraps the operation error with a stable context.
func GeneratedErrorsError(cause error) error {
	if cause == nil {
		return ErrGeneratedErrors
	}
	return errors.Join(ErrGeneratedErrors, cause)
}
