package tensor

import "fmt"

func Fused1590Log2AddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Sign()
	return r, nil
}
func Fused1590Log2AddSignName() string { return "log2_add_sign" }
