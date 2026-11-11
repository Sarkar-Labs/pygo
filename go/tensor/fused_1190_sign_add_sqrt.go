package tensor

import "fmt"

func Fused1190SignAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Sqrt()
	return r, nil
}
func Fused1190SignAddSqrtName() string { return "sign_add_sqrt" }
