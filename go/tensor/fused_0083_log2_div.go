package tensor

import "fmt"

func Fused0083Log2Div(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	return r, nil
}
func Fused0083Log2DivName() string { return "log2_div" }
