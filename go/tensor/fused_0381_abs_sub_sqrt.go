package tensor

import "fmt"

func Fused0381AbsSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Sqrt()
	return r, nil
}
func Fused0381AbsSubSqrtName() string { return "abs_sub_sqrt" }
