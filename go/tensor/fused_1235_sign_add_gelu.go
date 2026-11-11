package tensor

import "fmt"

func Fused1235SignAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.GELU()
	return r, nil
}
func Fused1235SignAddGeluName() string { return "sign_add_gelu" }
