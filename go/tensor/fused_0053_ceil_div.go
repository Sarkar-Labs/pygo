package tensor

import "fmt"

func Fused0053CeilDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	return r, nil
}
func Fused0053CeilDivName() string { return "ceil_div" }
