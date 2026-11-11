package tensor

import "fmt"

func Fused1011CeilSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Sqrt()
	return r, nil
}
func Fused1011CeilSubSqrtName() string { return "ceil_sub_sqrt" }
