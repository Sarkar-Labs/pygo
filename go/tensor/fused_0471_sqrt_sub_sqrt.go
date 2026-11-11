package tensor

import "fmt"

func Fused0471SqrtSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Sqrt()
	return r, nil
}
func Fused0471SqrtSubSqrtName() string { return "sqrt_sub_sqrt" }
