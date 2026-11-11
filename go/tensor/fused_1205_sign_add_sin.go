package tensor

import "fmt"

func Fused1205SignAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Sin()
	return r, nil
}
func Fused1205SignAddSinName() string { return "sign_add_sin" }
