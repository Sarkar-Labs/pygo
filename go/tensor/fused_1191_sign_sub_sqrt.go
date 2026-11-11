package tensor

import "fmt"

func Fused1191SignSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Sqrt()
	return r, nil
}
func Fused1191SignSubSqrtName() string { return "sign_sub_sqrt" }
