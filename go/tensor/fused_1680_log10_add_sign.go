package tensor

import "fmt"

func Fused1680Log10AddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Sign()
	return r, nil
}
func Fused1680Log10AddSignName() string { return "log10_add_sign" }
