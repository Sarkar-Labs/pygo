package tensor

import "fmt"

func Fused0693LogDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Sign()
	return r, nil
}
func Fused0693LogDivSignName() string { return "log_div_sign" }
