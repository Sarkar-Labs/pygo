package tensor

import "fmt"

func Fused1230SignAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Sign()
	r = r.Sign()
	return r, nil
}
func Fused1230SignAddSignName() string { return "sign_add_sign" }
