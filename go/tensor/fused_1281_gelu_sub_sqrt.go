package tensor

import "fmt"

func Fused1281GeluSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Sqrt()
	return r, nil
}
func Fused1281GeluSubSqrtName() string { return "gelu_sub_sqrt" }
