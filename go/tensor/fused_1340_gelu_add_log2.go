package tensor

import "fmt"

func Fused1340GeluAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Log2()
	return r, nil
}
func Fused1340GeluAddLog2Name() string { return "gelu_add_log2" }
