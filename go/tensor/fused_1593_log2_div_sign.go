package tensor

import "fmt"

func Fused1593Log2DivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Sign()
	return r, nil
}
func Fused1593Log2DivSignName() string { return "log2_div_sign" }
