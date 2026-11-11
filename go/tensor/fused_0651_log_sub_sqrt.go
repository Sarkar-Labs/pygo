package tensor

import "fmt"

func Fused0651LogSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Sqrt()
	return r, nil
}
func Fused0651LogSubSqrtName() string { return "log_sub_sqrt" }
