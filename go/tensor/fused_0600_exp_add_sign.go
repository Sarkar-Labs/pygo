package tensor

import "fmt"

func Fused0600ExpAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Sign()
	return r, nil
}
func Fused0600ExpAddSignName() string { return "exp_add_sign" }
