package tensor

import "fmt"

func Fused1236SignSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.GELU()
	return r, nil
}
func Fused1236SignSubGeluName() string { return "sign_sub_gelu" }
