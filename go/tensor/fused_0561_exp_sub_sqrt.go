package tensor

import "fmt"

func Fused0561ExpSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Sqrt()
	return r, nil
}
func Fused0561ExpSubSqrtName() string { return "exp_sub_sqrt" }
