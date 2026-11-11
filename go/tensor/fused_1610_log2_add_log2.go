package tensor

import "fmt"

func Fused1610Log2AddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Log2()
	return r, nil
}
func Fused1610Log2AddLog2Name() string { return "log2_add_log2" }
