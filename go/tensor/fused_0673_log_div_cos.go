package tensor

import "fmt"

func Fused0673LogDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Cos()
	return r, nil
}
func Fused0673LogDivCosName() string { return "log_div_cos" }
