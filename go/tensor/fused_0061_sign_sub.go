package tensor

import "fmt"

func Fused0061SignSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	return r, nil
}
func Fused0061SignSubName() string { return "sign_sub" }
