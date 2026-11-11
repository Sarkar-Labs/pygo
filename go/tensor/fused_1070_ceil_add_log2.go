package tensor

import "fmt"

func Fused1070CeilAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Log2()
	return r, nil
}
func Fused1070CeilAddLog2Name() string { return "ceil_add_log2" }
