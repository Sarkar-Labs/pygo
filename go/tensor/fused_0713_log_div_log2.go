package tensor

import "fmt"

func Fused0713LogDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Log2()
	return r, nil
}
func Fused0713LogDivLog2Name() string { return "log_div_log2" }
