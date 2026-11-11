package tensor

import "fmt"

func Fused1681Log10SubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Sign()
	return r, nil
}
func Fused1681Log10SubSignName() string { return "log10_sub_sign" }
