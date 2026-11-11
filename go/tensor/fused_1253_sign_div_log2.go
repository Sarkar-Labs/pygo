package tensor

import "fmt"

func Fused1253SignDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Log2()
	return r, nil
}
func Fused1253SignDivLog2Name() string { return "sign_div_log2" }
