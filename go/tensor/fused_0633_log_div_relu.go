package tensor

import "fmt"

func Fused0633LogDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.ReLU()
	return r, nil
}
func Fused0633LogDivReluName() string { return "log_div_relu" }
