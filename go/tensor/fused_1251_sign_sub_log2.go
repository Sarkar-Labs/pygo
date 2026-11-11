package tensor

import "fmt"

func Fused1251SignSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Log2()
	return r, nil
}
func Fused1251SignSubLog2Name() string { return "sign_sub_log2" }
